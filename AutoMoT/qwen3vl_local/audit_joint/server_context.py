"""Record the machine and local paths without guessing or copying server assets."""
import os
from pathlib import Path
import socket


def context(**paths):
    return dict(server_label=os.environ.get('AUTOMOT_SERVER','unspecified'),
        hostname=socket.gethostname(),project_root=str(Path(__file__).resolve().parents[2]),
        paths={k:dict(logical=str(Path(v).absolute()),resolved=str(Path(v).resolve()),
                      exists=Path(v).exists()) for k,v in paths.items() if v is not None},
        scope='Local execution/collection location only; cr02 and cee do not share assets. '
              'A user-supplied server label is not identity authentication or data admission.')
