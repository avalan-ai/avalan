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
from .head import COMMAND_DEFINITION as HEAD_COMMAND
from .jq import COMMAND_DEFINITION as JQ_COMMAND
from .ls import COMMAND_DEFINITION as LS_COMMAND
from .pdftoppm import COMMAND_DEFINITION as PDFTOPPM_COMMAND
from .pdftotext import COMMAND_DEFINITION as PDFTOTEXT_COMMAND
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
    WC_COMMAND,
    AWK_COMMAND,
    SED_COMMAND,
    JQ_COMMAND,
    PDFTOTEXT_COMMAND,
    PDFTOPPM_COMMAND,
    TESSERACT_COMMAND,
)
