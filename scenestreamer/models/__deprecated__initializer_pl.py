import lightning.pytorch as pl
import torch
from scenestreamer.models.initializer import SceneStreamerModel
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingWarmRestarts

from scenestreamer.dataset import constants
from scenestreamer.eval import metrics


class SceneStreamerInitializer(pl.LightningModule):
    def __init__(self, cfg):
        if "SEED" in cfg:
            pl.seed_everything(cfg.SEED)
            print("Everything is seeded to: ", cfg.SEED)

        super().__init__()
        self.cfg = cfg
        self.model_cfg = self.cfg.MODEL
        self.sampling_method = self.cfg.INITIALIZER.SAMPLING_METHOD

        self.decoder = SceneStreamerModel(config=self.model_cfg)

        self.save_hyperparameters()

        self.validation_outputs = []
        self.validation_ground_truth = []

        mmd_metrics = {}
        for word in ["", "_vehicle", "_pedestrian", "_cyclist"]:
            mmd_metrics.update(
                {
                    f"mmd_pos{word}": metrics.MMD(kernel_mul=1.0, kernel_num=1),
                    f"mmd_size{word}": metrics.MMD(kernel_mul=1.0, kernel_num=1),
                    f"mmd_head{word}": metrics.MMD(kernel_mul=1.0, kernel_num=1),
                    f"mmd_vel{word}": metrics.MMD(kernel_mul=1.0, kernel_num=1),
                }
            )
        for k, v in mmd_metrics.items():
            self.register_module(k, v)
        self.mmd_metrics_keys = list(mmd_metrics.keys())

    def autoregressive_generate(self, *args, **kwargs):
        return self.decoder.autoregressive_generate(*args, **kwargs)

    def forward(self, batch_dict):
        forward_ret_dict = self.decoder(batch_dict)
        return forward_ret_dict

    def get_loss(self, data_dict, gt_dict, forward_ret_dict):
        loss, tb_dict, disp_dict = self.decoder.get_loss(data_dict, gt_dict, forward_ret_dict)
        return loss, tb_dict, disp_dict

    def training_step(self, batch, batch_idx):
        data_dict, gt_dict = batch
        forward_ret_dict = self(data_dict)
        loss, tb_dict, disp_dict = self.get_loss(data_dict, gt_dict, forward_ret_dict)
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in tb_dict.items()},
            batch_size=data_dict["encoder/agent_feature"].shape[0],
        )
        self.log('monitoring_step', float(self.global_step))
        return loss

    def validation_step(self, batch, batch_idx, condition_on_sdc=None):
        data_dict, gt_dict = batch

        if condition_on_sdc is None:
            condition_on_sdc = self.cfg.INITIALIZER.CONDITION_ON_SDC

        num_v = data_dict["encoder/agent_valid_mask"][:, 0].sum(-1).max()
        ret = self.autoregressive_generate(
            data_dict=batch,
            num_v=num_v,
            deterministic_state=True,
            temperature=self.cfg.INITIALIZER.TEMPERATURE,
            sampling_method=self.cfg.INITIALIZER.SAMPLING_METHOD,
            topk=self.cfg.INITIALIZER.TOPK,
            topp=self.cfg.INITIALIZER.TOPP,
            use_nature_probability=self.cfg.INITIALIZER.USE_NATURE_PROBABILITY,
            condition_on_sdc=condition_on_sdc
        )
        self.log("eval/intersection_count", ret["intersection_count"], on_step=True)
        self.log("eval/intersection_total_count", ret["total_count"], on_step=True)
        self.log("eval/intersection_rate", ret["intersection_count"] / ret["total_count"], on_step=True)

        B = ret["sampled_pos"].shape[0]
        for i in range(B):

            pos_target = data_dict["encoder/agent_position"][i, 0, ..., :2]
            vel_target = data_dict["encoder/agent_feature"][i, 0, ..., 4:6]  #* constants.VELOCITY_XY_RANGE
            head_target = data_dict["encoder/agent_feature"][i, 0, ..., 3:4]  #* constants.HEADING_RANGE
            size_target = data_dict["encoder/agent_feature"][i, 0, ..., 6:8]  # * constants.SIZE_RANGE
            valid_mask = data_dict["encoder/agent_valid_mask"][i, 0]
            if condition_on_sdc:
                valid_mask[gt_dict["sdc_index"][i]] = False  # Mask out SDC in target

            pos_target = pos_target[valid_mask]
            vel_target = vel_target[valid_mask]
            head_target = head_target[valid_mask]
            size_target = size_target[valid_mask]

            num_target = len(pos_target)

            actor_type = data_dict["decoder/actor_type"][i][valid_mask]
            for suffix, mask in {
                    "": pos_target.new_ones(num_target, dtype=bool),
                    "_vehicle": actor_type == 1,
                    "_pedestrian": actor_type == 2,
                    "_cyclist": actor_type == 3,
            }.items():
                if not mask.any():
                    continue
                self.get_submodule(f"mmd_pos{suffix}").update(
                    source=ret["sampled_pos"][i, :num_target, :2][mask], target=pos_target[mask]
                )
                self.get_submodule(f"mmd_vel{suffix}").update(
                    source=ret["sampled_vel"][i, :num_target][mask], target=vel_target[mask]
                )
                self.get_submodule(f"mmd_head{suffix}").update(
                    source=ret["sampled_head"][i, :num_target].unsqueeze(-1)[mask], target=head_target[mask]
                )
                self.get_submodule(f"mmd_size{suffix}").update(
                    source=ret["sampled_size"][i, :num_target, :2][mask], target=size_target[mask]
                )

    def on_validation_epoch_end(self):
        for k in self.mmd_metrics_keys:
            self.log(f'eval/{k}', self.get_submodule(k))

    def on_validation_start(self):
        torch.cuda.empty_cache()

    def on_validation_end(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        opt_cfg = self.cfg.OPTIMIZATION

        if opt_cfg.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(
                [each[1] for each in self.named_parameters()],
                lr=opt_cfg.LR,
                weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            )
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0), betas=(0.9, 0.95)
            )
        else:
            assert False

        if opt_cfg.get('SCHEDULER', None) == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=2,
                T_mult=1,
                eta_min=max(1e-2 * opt_cfg.LR, 1e-6),
                last_epoch=-1,
            )
        elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':

            def lr_lbmd(cur_epoch):
                cur_decay = 1
                for decay_step in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20]):
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * opt_cfg.LR_DECAY
                return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

            scheduler = LambdaLR(optimizer, lr_lbmd)

        elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=opt_cfg.LR_CLIP / opt_cfg.LR,
                total_iters=opt_cfg.NUM_EPOCHS,
            )
        else:
            scheduler = None

        return {
            "optimizer": optimizer,

            # PZH NOTE: The scheduler step will be added 1 after each epoch.
            "lr_scheduler": scheduler
        }

    def generate_from_scratch(
        self, input_tuple, num_v, num_p, num_c, temperature, angle_limit_in_deg, compress_step=None
    ):
        data_dict, gt_dict = input_tuple

        self.eval()
        with torch.no_grad():
            ret, new_feat, actor_type_list = self.decoder.get_map_and_start_token(
                data_dict,
                num_v=num_v,
                num_p=num_p,
                num_c=num_c,
                temperature=temperature,
                angle_limit_in_deg=angle_limit_in_deg,
            )

        sampled = {k: torch.cat([v[k] for v in ret], dim=0).squeeze(1) for k in ret[0].keys()}

        # B, T, N, _ = data_dict["encoder/agent_feature"].shape

        N, _ = sampled["sampled_pos"].shape
        B = 1

        fake_pos = data_dict["encoder/map_position"].new_zeros([B, 91, N, 2])

        fake_pos[:, 4] = sampled["sampled_pos"]

        actor_type = torch.tensor(actor_type_list, device=self.device).reshape(B, N)

        # actor_feature = torch.stack(new_feat, dim=2)

        actor_feature = data_dict["encoder/map_feature"].new_zeros([B, 91, N, constants.AGENT_STATE_DIM])
        actor_feature[:, :5] = torch.stack(new_feat, dim=2)

        actor_feature = data_dict["encoder/map_feature"].new_zeros([B, 91, N, constants.AGENT_STATE_DIM])
        actor_feature[:, :5] = torch.stack(new_feat, dim=2)

        fake_data = {
            "encoder/map_feature": data_dict["encoder/map_feature"],
            "encoder/map_position": data_dict["encoder/map_position"],
            "encoder/map_feature_valid_mask": data_dict["encoder/map_feature_valid_mask"],
            "encoder/traffic_light_feature": data_dict["encoder/traffic_light_feature"],
            "encoder/traffic_light_position": data_dict["encoder/traffic_light_position"],
            "encoder/traffic_light_valid_mask": data_dict["encoder/traffic_light_valid_mask"],
            "encoder/agent_feature": actor_feature,
            "encoder/agent_position": fake_pos,
            "encoder/agent_valid_mask": data_dict["encoder/map_feature_valid_mask"].new_ones([B, 91, N]),
            "decoder/actor_type": actor_type
        }

        gt_dict["current_time_index"].fill_(5)

        ar_ret = self.autoregressive_generate((fake_data, gt_dict), compress_step=compress_step)

        return ar_ret, fake_data, sampled
