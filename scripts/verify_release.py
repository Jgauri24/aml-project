"""Read-only integrity check of packaged artifacts; does not fit or score models."""
import json
from pathlib import Path
import pandas as pd
from src.config import ROOT
from src.release.evaluate import RELEASE, digest, verify_freeze


def main():
    verify_freeze()
    manifest=json.loads((RELEASE/'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path=RELEASE/relative
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    for scope,origins,count in [('development',[1801,1829,1857,1885],26880),('final',[1913],6720)]:
        p=pd.read_parquet(RELEASE/scope/'predictions.parquet')
        assert sorted(p.origin.unique())==origins
        assert p.groupby('model').size().eq(count).all()
        assert set(p.model)=={'seasonal_naive','kalman','lightgbm','deepar'}
        assert not p.duplicated(['model','series_id','origin','horizon']).any()
        assert (p.target_day_num == p.origin+p.horizon).all()
        if scope=='development': assert p.target_day_num.max()<=1913
        else: assert p.target_day_num.between(1914,1941).all()
    print(f"Verified {len(manifest['files'])} file hashes and both forecast grids; no fitting or scoring performed")

if __name__=='__main__':
    main()
