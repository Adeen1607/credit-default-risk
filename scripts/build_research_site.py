from pathlib import Path
import json, re, base64, shutil, hashlib, html, subprocess, os
import nbformat
from nbconvert import HTMLExporter
import markdown
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'credit-default-risk': ('Calibrated Credit Default Risk', 'How does a calibrated baseline trade off missed defaults against false alerts under an explicit cost assumption?', 'UCI Default of Credit Card Clients dataset.', 'Training-only imputation and scaling precede class-weighted logistic regression. Five-fold sigmoid calibration estimates probability corrections. Validation data selects a threshold under an assumed 5:1 error-cost ratio; test data measures the locked decision.', 'Assumed error costs are not observed financial losses. Historical Taiwan records do not establish suitability for current Canadian lending. Demographic variables need fairness review before any operational use.')}

CSS = '''body{margin:0;background:#f6f8fb;color:#172338;font:17px/1.7 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:40px 28px 80px}header{background:#10283e;color:white;padding:48px 28px}header>div{max-width:1050px;margin:auto}h1{font-size:clamp(30px,5vw,46px);line-height:1.2}h2{margin-top:2.2em;color:#153f5e}h3{color:#245674}a{color:#1163a1}header a{color:#cae8ff}nav{display:flex;gap:24px;flex-wrap:wrap}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px 14px;text-align:left;border:1px solid #ccd5df}th{background:#e4edf5}pre{overflow:auto;background:#e9eef4;padding:18px;border-radius:5px}code{font-size:14px}figure{margin:30px 0}figcaption{font-size:14px;color:#506379}.meta{font-size:14px;letter-spacing:.08em;text-transform:uppercase}details{margin:18px 0;padding:12px;border:1px solid #cdd8e2}summary{cursor:pointer}footer{margin-top:50px;font-size:14px;color:#506379} @media(max-width:600px){main{padding:24px 16px}table{display:block;overflow:auto}header{padding:30px 16px}}'''

def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def md(s):return markdown.markdown(s,extensions=['tables','fenced_code','toc'])
def page(title,body):return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><link rel="stylesheet" href="style.css"></head><body><header><div><p class="meta">Mohammed Adeen Shaik · Research and coursework</p><h1>{html.escape(title)}</h1><nav><a href="index.html">Overview</a><a href="report.html">Research report</a><a href="notebooks.html">Explained notebooks</a><a href="https://github.com/Adeen1607/{CURRENT}">Source repository</a></nav></div></header><main>{body}<footer>Evidence dates and limitations are documented in the report. Static pages display saved outputs; they do not execute Python, R, Streamlit or API calls.</footer></main></body></html>'

def guide(source):
    notes=[]
    rules=[
      (r'read_csv|read_excel','Load the source table','A DataFrame is a spreadsheet-like table in memory. The path must point to the documented input file; loading a file does not validate its contents.'),
      (r'\.iloc\[|\.values|reshape','Select values and shape inputs','`iloc` selects by column position. Check the source schema because a changed column order can select the wrong variable. `reshape(-1, 1)` makes a one-column table expected by a model.'),
      (r'groupby|\.agg\(|value_counts|\.sum\(','Summarize observations','Grouping collects rows with the same label. Counts and sums produce totals; a mean of a binary 0/1 indicator gives a rate. Always identify the denominator and keep missing categories visible.'),
      (r'train_test_split','Separate learning and assessment','Training rows teach the model; validation rows help choose settings. Test rows estimate performance after those choices are fixed. `stratify` preserves label proportions and `random_state` makes the split repeatable.'),
      (r'fit_transform|StandardScaler|MinMaxScaler','Understand feature scaling','Scaling changes numerical units so large values do not dominate. `fit` learns statistics and `transform` applies them. Those statistics should be learned only from training rows, including inside each cross-validation fold.'),
      (r'quantile|IQR|lower_bound|upper_bound','Inspect extreme values','Quartiles describe the middle half of a distribution. Values beyond an IQR fence are unusual, not automatically erroneous. Replacing extreme fraud signals can remove information and must be justified.'),
      (r'cross_val_score|StratifiedKFold','Compare across folds','Cross-validation rotates the held-out subset. A mean summarizes the selected metric across folds; the spread shows variation. Preprocessing belongs inside each fold and repeated related records may require group-aware splitting.'),
      (r'\.fit\(','Fit the model','Fitting learns a relationship from the supplied predictors and labels. A good training fit is not evidence of performance on unseen observations.'),
      (r'predict_proba|\.predict\(','Produce scores or predictions','A prediction is the model output for a row. A probability score ranks risk; applying a decision threshold trades missed positives against false alerts. The threshold must be chosen before looking at test outcomes.'),
      (r'precision|recall|f1_score|roc_curve|log_loss|\.score\(','Interpret evaluation','Accuracy counts all correct labels; recall measures captured positives and precision measures how many alerts are correct. Use the metric appropriate to the task. R-squared measures explained variation, not classification accuracy, and a training score is not a test score.'),
      (r'pd.cut|qcut','Create comparable bands','`cut` uses fixed boundaries; `qcut` attempts equal-sized quantile groups. Tied values can reduce the number of groups. Compare both observed rates and the number of records in each band.'),
      (r'plt\.|pyplot\.|figure|sns\.|\.plot\(','Read the chart','The axis labels identify units and the grouping defines what is compared. A visual association does not establish cause. Rate charts need denominators; truncated axes can exaggerate small differences.'),
      (r'corr\(','Explore association','Correlation summarizes association and can miss nonlinear relationships. It does not identify cause or prove a variable is a reliable predictor.'),
      (r'FAISS|Embeddings|split_text|text_splitter','Prepare retrieval evidence','Embeddings turn text into numerical representations for similarity search. Chunking limits the text retrieved at one time; overlap preserves boundary context. Similarity is not a truth score.'),
      (r'RetrievalQA|LLMChain|StuffDocumentsChain|qa_chain|PromptTemplate|llm =','Connect evidence and generation','The retriever supplies evidence to the prompt and a language model writes an answer. A grounding instruction alone cannot guarantee factuality. Here titles and authors cannot support detailed article summaries.'),
      (r'pickle|joblib','Save a model artifact','Serialization stores fitted objects for later use. Load model files only from a trusted source; pickle-like formats can execute code when loaded.'),
      (r'random.randint|random.choice|default_rng','Generate simulated observations','These random values are demonstration data. A fixed seed makes a rerun repeatable, but does not turn simulated observations into real survey evidence.'),
    ]
    for pattern,title,note in rules:
      if re.search(pattern,source):notes.append(f'**{title}.** {note}')
    return '\n\n'.join(notes[:3]) or 'This cell prepares named objects used by later steps. Read the assignments from right to left: the expression is evaluated, then stored under the name on the left. Run cells in order so those names exist.'

def notebook(p,conf,report_summary,figs):
    nb=nbformat.read(p,as_version=4)
    # Idempotent editorial cells carry a tag and can be rebuilt without duplication.
    cells=[c for c in nb.cells if 'research-editorial' not in c.metadata.get('tags',[])]
    intro=nbformat.v4.new_markdown_cell(f'# {conf[0]}\n\n**Author:** Mohammed Adeen Shaik\n\n## Research question\n\n{conf[1]}\n\n## Data and evidence\n\n{conf[2]}\n\n{report_summary}\n\n## Reading guide\n\nRead each explanation before its code cell. Run from top to bottom in the documented working directory. A saved chart is an output snapshot, not proof that this notebook was rerun today. The final section collects findings and visual evidence.\n\n## Method\n\n{conf[3]}\n\n## Interpretation limits\n\n{conf[4]}')
    intro.metadata.tags=['research-editorial']
    new=[intro]
    for i,c in enumerate(cells):
      if c.cell_type=='code' and c.source.strip():
        g=nbformat.v4.new_markdown_cell(f'### Guide to code step {i+1}\n\n'+guide(c.source));g.metadata.tags=['research-editorial'];new.append(g)
      if c.cell_type=='code' and not c.source.strip() and not c.outputs:continue
      new.append(c)
    ending='## Findings and final visual results\n\n'+report_summary+'\n\n'+conf[4]
    if figs:
      ending+='\n\nThe following figures are copied from saved notebook outputs or the documented evaluation artifacts. Their source and evidence status are given in the research report.\n'
      end=nbformat.v4.new_markdown_cell(ending);end.metadata.tags=['research-editorial'];new.append(end)
      for f,cap in figs[-4:]:
        c=nbformat.v4.new_markdown_cell(f'**{cap}**\n\n![{cap}](attachment:{f.name})');c['attachments']={f.name:{'image/png':base64.b64encode(f.read_bytes()).decode()}};c.metadata.tags=['research-editorial'];new.append(c)
    else:
      ending+='\n\nNo numerical result chart is available yet. Follow the reproduction instructions and publish the resulting outputs before making empirical claims.'
      end=nbformat.v4.new_markdown_cell(ending);end.metadata.tags=['research-editorial'];new.append(end)
    nb.cells=new
    nb.nbformat_minor=5
    nb.metadata.setdefault('kernelspec',{'display_name':'Python 3','language':'python','name':'python3'})
    nbformat.validate(nb);nbformat.write(nb,p)
    exp=HTMLExporter(template_name='lab');exp.exclude_input_prompt=True;exp.exclude_output_prompt=True
    rendered,_=exp.from_notebook_node(nb)
    style='<style>body{font-family:system-ui,sans-serif}#research-nav{padding:18px 30px;background:#10283e;color:white}#research-nav a{color:#cae8ff;margin-right:24px}.jp-Notebook{max-width:1150px;margin:auto}.jp-OutputArea-output{overflow-x:auto}.jp-MarkdownOutput{line-height:1.7}</style>'
    rendered=rendered.replace('</head>',style+'</head>').replace('<body>','<body><nav id="research-nav"><a href="../index.html">Project overview</a><a href="../report.html">Research report</a><a href="../notebooks.html">All notebooks</a></nav>')
    return rendered

def extract_figs(repo,site):
    figs=[]
    for p in sorted(repo.rglob('*.ipynb')):
      nb=json.loads(p.read_text())
      items=[]
      for i,c in enumerate(nb['cells']):
        if 'research-editorial' in c.get('metadata',{}).get('tags',[]):continue
        for o in c.get('outputs',[]):
          data=o.get('data',{}).get('image/png')
          if data:
            b=base64.b64decode(''.join(data) if isinstance(data,list) else data);items.append((i,b))
      for i,b in items[-3:]:
        f=site/'figures'/f'{p.stem.replace(" ","_")}-cell-{i+1}.png';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
        figs.append((f,f'Figure from {p.name}, original code cell {i+1}'))
    return figs

def run():
 global CURRENT
 for repo in [ROOT]:
  if repo.name not in CONFIG:continue
  CURRENT=repo.name;conf=CONFIG[CURRENT];site=repo/'research';site.mkdir(exist_ok=True)
  figs=extract_figs(repo,site)
  # Existing source-embedded screenshots are historical report evidence.
  import zipfile
  for p in sorted(repo.glob('*.docx')):
   if CURRENT=='GCP-ETL---Apache-Kafka':continue # Avoid propagating any credential-bearing screenshot.
   with zipfile.ZipFile(p) as z:
    images=[n for n in z.namelist() if n.startswith('word/media/') and n.lower().endswith('.png')]
    for i,n in enumerate(images[-3:]):
     f=site/'figures'/f'archival-{i+1}.png';f.parent.mkdir(exist_ok=True);f.write_bytes(z.read(n));figs.append((f,f'Historical figure {Path(n).name} from {p.name}; not a refreshed experiment'))
  for artifact in sorted((repo/'reports/figures').glob('*.png')):
   target=site/'figures'/artifact.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(artifact,target)
   figs.append((target,'Reproduced aggregate figure: '+artifact.name))
  for artifact in sorted((repo/'reports/results').glob('*.png')):
   target=site/'figures'/artifact.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(artifact,target)
   figs.append((target,'Held-out evaluation figure: '+artifact.name))
  resultfile=repo/'reports'/'research-findings.md' 
  summary=resultfile.read_text() if resultfile.exists() else 'Evidence status: historical outputs or implementation only. Numerical findings are limited to the original artifacts; no new model performance is claimed.'
  manifest={'repository':f'Adeen1607/{CURRENT}','review_date':'2026-10-11','evidence_note':summary,'inputs':[]}
  for p in repo.rglob('*'):
   if p.is_file() and '.git' not in p.parts and not any(x in p.parts for x in ['research','__pycache__','artifacts','processed','raw']):
    if p.suffix in ['.csv','.xlsx','.ipynb','.R','.json']:
     manifest['inputs'].append({'path':str(p.relative_to(repo)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

  original=repo/'README.md';readme=original.read_text() if original.exists() else f'# {conf[0]}\n\n{conf[2]}\n'
  links=[]
  for p in sorted(repo.rglob('*.ipynb')):
   if 'research' in p.parts:continue
   outfile='notebooks/'+re.sub(r'[^a-zA-Z0-9_-]','_',p.stem)+'.html'
   ownfigs=[x for x in figs if p.name in x[1] or 'aggregate figure' in x[1] or 'Held-out evaluation' in x[1]]
   rendered=notebook(p,conf,summary,ownfigs)
   write(site/outfile,rendered);links.append((p,outfile))
  for item in manifest['inputs']:
   item['sha256']=hashlib.sha256((repo/item['path']).read_bytes()).hexdigest()
  write(site/'evidence.json',json.dumps(manifest,indent=2)+'\n')
  report=f'# {conf[0]}\n\n**Author:** Mohammed Adeen Shaik  \n**Review date:** 11 October 2026\n\n## Research question\n\n{conf[1]}\n\n## Data and provenance\n\n{conf[2]}\n\nSource materials are linked from the repository README. Input fingerprints are recorded in [the evidence manifest](evidence.json) so a reader can identify the exact committed artifacts used in this review.\n\n## Method and analytical rationale\n\n{conf[3]}\n\n## Findings and interpretation\n\n{summary}\n\n## Limitations\n\n{conf[4]}\n\n## Reproduction\n\nUse the setup commands and input contract in the [repository README](https://github.com/Adeen1607/{CURRENT}#readme). Open the [notebook catalogue](notebooks.html) for annotated code and saved outputs. Keep raw records and model binaries local; public reports contain aggregate results or source-embedded figures.\n\n## Final visual results\n\n'
  if figs:
   for f,cap in figs:report+=f'![{cap}]({f.relative_to(site)})\n\n*{cap}.*\n\n'
  else:report+='No verified numerical result figure is available in the current evidence. This status is shown explicitly rather than substituting fabricated data.\n'
  write(site/'report.md',report);write(site/'report.html',page(conf[0],md(report)))
  catalogue='<h2>Annotated notebook catalogue</h2><p>Each notebook explains the research question, data requirements, difficult operations and interpretation limits. Saved outputs are rendered without executing external APIs.</p>'
  if links:
   catalogue+='<ul>'+''.join(f'<li><a href="{url}">{html.escape(p.name)}</a> · <a href="https://github.com/Adeen1607/{CURRENT}/blob/{subprocess.check_output(["git","-C",str(repo),"branch","--show-current"],text=True).strip() or "main"}/{str(p.relative_to(repo)).replace(" ","%20")}">Download / inspect source</a></li>' for p,url in links)+'</ul>'
  else:catalogue+='<p>This repository has no Python notebook. Its report explains the available source or archived materials.</p>'
  write(site/'notebooks.html',page(conf[0],catalogue))
  overview=f'<h2>The study</h2><p>{html.escape(conf[1])}</p><p>{html.escape(conf[2])}</p><h2>What the evidence shows</h2>{md(summary)}<h2>Read the work</h2><p><a href="report.html">Research report and final result figures</a> · <a href="notebooks.html">Explained notebooks</a></p><h2>Interpretation</h2><p>{html.escape(conf[4])}</p>'
  write(site/'index.html',page(conf[0],overview));write(site/'style.css',CSS);write(site/'.nojekyll','')
  # Publishing a pre-rendered site keeps notebooks with missing raw data readable.
  branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip() or 'main'
  workflow='''name: Publish research notebooks
on:
  push:
    branches: [BRANCH]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: research-pages
  cancel-in-progress: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: research
      - id: deployment
        uses: actions/deploy-pages@v4
'''.replace('BRANCH',branch)
  if CURRENT not in ['Adeen1607','GUI_Beginner_Project','SENG8081-24S-Sec1','ML-AUDIO_CLASSIFICATION']:
   write(repo/'.github/workflows/research-pages.yml',workflow)
  readme=re.sub(r'\n<!-- research-edition:start -->.*?<!-- research-edition:end -->\n?','\n',readme,flags=re.S)
  readme+='\n<!-- research-edition:start -->\n## Research edition\n\n'+f'[Research report and final visual results](research/report.md) · [Browser-ready notebook catalogue](research/notebooks.html)\n\n'+summary+'\n\n'+f'The `research/` folder contains a professional static reading edition. The Pages workflow deploys it after GitHub Pages is enabled with **GitHub Actions** as the publishing source. Intended URL: `https://adeen1607.github.io/{CURRENT}/`. Deployment status must be checked in Actions before treating that address as live.\n\n<!-- research-edition:end -->\n'
  write(original,readme)
  print(CURRENT,len(links),'notebooks',len(figs),'figures',flush=True)

if __name__=='__main__':run()
