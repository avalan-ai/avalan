from .awk import COMMAND_DEFINITION as AWK_COMMAND
from .base import (
    NormalizedPath as NormalizedPath,
)
from .base import (
    NormalizedWorkspace as NormalizedWorkspace,
)
from .base import (
    ShellCommandDefinition as ShellCommandDefinition,
)
from .base import (
    ShellCommandPolicyContext as ShellCommandPolicyContext,
)
from .base import (
    ShellDependencyGroup as ShellDependencyGroup,
)
from .cat import COMMAND_DEFINITION as CAT_COMMAND
from .file import COMMAND_DEFINITION as FILE_COMMAND
from .find import COMMAND_DEFINITION as FIND_COMMAND
from .head import COMMAND_DEFINITION as HEAD_COMMAND
from .jq import COMMAND_DEFINITION as JQ_COMMAND
from .kill import COMMAND_DEFINITION as KILL_COMMAND
from .ls import COMMAND_DEFINITION as LS_COMMAND
from .lsof import COMMAND_DEFINITION as LSOF_COMMAND
from .nl import COMMAND_DEFINITION as NL_COMMAND
from .pdfinfo import COMMAND_DEFINITION as PDFINFO_COMMAND
from .pdfplumber import COMMAND_DEFINITION as PDFPLUMBER_COMMAND
from .pdftoppm import COMMAND_DEFINITION as PDFTOPPM_COMMAND
from .pdftotext import COMMAND_DEFINITION as PDFTOTEXT_COMMAND
from .pgrep import COMMAND_DEFINITION as PGREP_COMMAND
from .ps import COMMAND_DEFINITION as PS_COMMAND
from .pypdf import COMMAND_DEFINITION as PYPDF_COMMAND
from .reportlab import COMMAND_DEFINITION as REPORTLAB_COMMAND
from .rg import COMMAND_DEFINITION as RG_COMMAND
from .sed import COMMAND_DEFINITION as SED_COMMAND
from .tail import COMMAND_DEFINITION as TAIL_COMMAND
from .tesseract import COMMAND_DEFINITION as TESSERACT_COMMAND
from .wc import COMMAND_DEFINITION as WC_COMMAND

SHELL_COMMANDS = (
    RG_COMMAND,
    HEAD_COMMAND,
    TAIL_COMMAND,
    LS_COMMAND,
    CAT_COMMAND,
    NL_COMMAND,
    PGREP_COMMAND,
    PS_COMMAND,
    LSOF_COMMAND,
    KILL_COMMAND,
    FILE_COMMAND,
    FIND_COMMAND,
    WC_COMMAND,
    AWK_COMMAND,
    SED_COMMAND,
    JQ_COMMAND,
    PDFINFO_COMMAND,
    PDFTOTEXT_COMMAND,
    PDFTOPPM_COMMAND,
    REPORTLAB_COMMAND,
    PDFPLUMBER_COMMAND,
    PYPDF_COMMAND,
    TESSERACT_COMMAND,
)
