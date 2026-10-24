"""Hash the exact bytes decoded at consumption, using frozen route receipts."""
import hashlib
import json
from pathlib import Path
from qwen3vl_local.audit_joint.abc_delivery import digest


def checked_bytes(path,expected):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('consumed_source_changed: '+str(path))
    return raw


class Sources:
    def __init__(self,directory,root,expected_identity=None,expected_manifest_sha256=None):
        self.directory=Path(directory);self.root=Path(root).resolve()
        path=self.directory/'manifest.json'
        raw=checked_bytes(path,expected_manifest_sha256) if expected_manifest_sha256 else path.read_bytes()
        self.manifest=json.loads(raw)
        if expected_identity is not None and self.manifest['identity']!=expected_identity:
            raise ValueError('consumed_dataset_changed')
        from .contracts import source_identity
        if self.manifest['code']!=source_identity(data_only=True):raise ValueError('consumed_data_contract_changed')
        self.receipts={}

    def receipt(self,record):
        key=digest([record['scenario'],record['route_id']])
        # Cache immutable receipt bytes after verifying against the admitted
        # manifest. Every sensor read below still hashes new bytes, never mtime.
        if key not in self.receipts:
            raw=checked_bytes(self.directory/'routes'/key/'receipt.json',self.manifest['routes'][key])
            self.receipts[key]=json.loads(raw)
        return self.receipts[key]

    def read(self,record,route,kind,suffix):
        expected_route=self.root/record['scenario']/record['route_id']
        if Path(route).resolve()!=expected_route.resolve():raise ValueError('consumed_route_root_changed')
        name=f'{record["scenario"]}/{record["route_id"]}/{kind}/{record["frame"]:04d}.{suffix}'
        value=self.receipt(record)['bindings'][name]
        if kind=='rgb' and value!=record['image_sha256']:raise ValueError('consumed_image_binding_changed')
        return checked_bytes(self.root/name,value),value

    def proof(self,record):
        receipt=self.receipt(record)
        return {kind:receipt['bindings'][f'{record["scenario"]}/{record["route_id"]}/{kind}/{record["frame"]:04d}.{ext}']
                for kind,ext in (('metas','pkl'),('rgb','jpg'),('lidar','laz'))}
