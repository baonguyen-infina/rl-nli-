"""Plot internal training diagnostics outside the test-only paper artifacts."""
import argparse
import csv
from pathlib import Path
from common.io import RESULTS,ROOT

OUTPUT=ROOT/'artifacts/analysis/figures/training.pdf'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',default=['D16','D18'])
    a=ap.parse_args();fig,axes=plt.subplots(1,2,figsize=(7,2.7))
    found=False
    for rid in a.runs:
        for folder in sorted(RESULTS.glob(rid+'_*_s*')):
            p=folder/'train_log.csv'
            if not p.exists():continue
            rows=[r for r in csv.DictReader(p.open()) if r.get('reward')]
            if not rows:continue
            found=True
            for ax,key in zip(axes,['reward','frac_reward_zero_std']):
                values=[float(r[key]) for r in rows]
                smooth=[sum(values[max(0,i-19):i+1])/len(values[max(0,i-19):i+1]) for i in range(len(values))]
                ax.plot(range(1,len(smooth)+1),smooth,label=folder.name.split('_')[0]+' '+folder.name.rsplit('_',1)[-1],lw=1)
                ax.set_xlabel('Optimizer step')
    if not found:raise SystemExit('No recorded curves')
    axes[0].set_ylabel('Training reward (20-step mean)')
    axes[1].set_ylabel('Groups with zero reward variance')
    axes[1].set_ylim(0,1);axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();out=OUTPUT;out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,bbox_inches='tight');print(out)

if __name__=='__main__':main()
