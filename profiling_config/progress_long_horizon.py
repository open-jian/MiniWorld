import re,json,time
from pathlib import Path
root=Path('/data2/jian/output/Miniworld/DROID_long_horizon')
print('Completed:',len(list((root/'episodes').glob('*/summary.json'))),'/50')
for p in sorted(list((root/'logs').glob('worker_*.log'))+list((root/'logs').glob('queue_*.log'))):
 text=p.read_text();done=text.count('[DONE]');m=re.findall(r'latent_frames=(\d+)/(\d+) \| step=(\d+)/(\d+)',text)
 print(p.stem,'completed',done,'current latent / step',m[-1] if m else 'loading', 'ERROR' if 'Traceback' in text else '')
