import pathlib

from metadrive.scenario import ScenarioDescription as SD
from scenarionet.builder.utils import try_generating_summary
from scenarionet.common_utils import save_summary_anda_mapping

FILE_ROOT = pathlib.Path(".")
if __name__ == '__main__':
    summary = try_generating_summary(FILE_ROOT.absolute())
    save_summary_anda_mapping(
        summary_file_path=FILE_ROOT / SD.DATASET.SUMMARY_FILE,
        mapping_file_path=FILE_ROOT / SD.DATASET.MAPPING_FILE,
        summary=summary,
        mapping={k: "."
                 for k in summary}
    )
