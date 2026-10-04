const $ = id => document.getElementById(id);
let selected;
function choose(file){
  if(!file)return;
  selected=file;$('filename').textContent=file.name;
  if($('preview').dataset.url)URL.revokeObjectURL($('preview').dataset.url);
  const url=URL.createObjectURL(file);$('preview').src=url;$('preview').dataset.url=url;
  $('preview').hidden=false;$('upload-prompt').hidden=true;$('error').hidden=true;
  $('result').hidden=true;$('inspection').hidden=true;$('empty').hidden=false;
}
$('file').addEventListener('change',e=>choose(e.target.files[0]));
for(const event of ['dragenter','dragover'])$('drop').addEventListener(event,e=>{e.preventDefault();$('drop').classList.add('drag')});
for(const event of ['dragleave','drop'])$('drop').addEventListener(event,e=>{e.preventDefault();$('drop').classList.remove('drag')});
$('drop').addEventListener('drop',e=>choose(e.dataTransfer.files[0]));
function list(id,items){$(id).replaceChildren(...items.map(text=>{const li=document.createElement('li');li.textContent=text;return li}))}
function featureRow(name,value){const row=document.createElement('div'),label=document.createElement('span'),val=document.createElement('b');label.textContent=name.replaceAll('_',' ');val.textContent=value.toFixed(4);row.append(label,val);return row}
$('upload').addEventListener('submit',async e=>{
  e.preventDefault();$('error').hidden=true;
  if(!selected){$('error').textContent='Choose an image first.';$('error').hidden=false;return}
  $('analyze').disabled=true;$('analyze').textContent='Extracting features & analyzing…';
  $('result').hidden=true;$('inspection').hidden=true;
  try{
    const form=new FormData();form.append('image',selected);
    const response=await fetch('/api/analyze',{method:'POST',body:form});const data=await response.json();
    if(!response.ok)throw new Error(data.error||'Analysis failed.');
    $('empty').hidden=true;$('result').hidden=false;$('inspection').hidden=false;
    const review=data.inference.status==='needs_review';
    $('prediction').textContent=review?'Possible '+data.prediction:data.prediction;$('confidence').textContent=(data.confidence*100).toFixed(1)+'%';
    $('status').textContent=review?'Uncertain result · retake image or seek expert review':'Supported candidate · confirm in the field';$('status').className='badge'+(review?' warn':'');
    $('probabilities').replaceChildren(...Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).map(([name,p])=>{
      const row=document.createElement('div');row.className='prob-row';const title=document.createElement('span');title.textContent=name;
      const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('span');fill.style.width=(p*100)+'%';bar.append(fill);
      const value=document.createElement('span');value.textContent=(p*100).toFixed(0)+'%';row.append(title,bar,value);return row;
    }));
    $('pest').textContent='Separate pest detector: '+(data.pest_probability*100).toFixed(1)+'% probability of spider-mite damage. '+data.pest_scope;
    const rec=data.inference.recommendation;list('treatment',rec.treatment);list('prevention',rec.prevention);
    $('provider').textContent=rec.provider+(rec.version?' · '+rec.version:'');$('advice-scope').textContent=rec.scope;
    $('sources').replaceChildren(...rec.sources.map(s=>{const a=document.createElement('a');a.href=s.url;a.textContent=s.title+' ↗';a.target='_blank';a.rel='noopener';return a}));
    for(const key of ['before','after','mask'])$(key).src=data.images[key];
    $('features').replaceChildren(...Object.entries(data.features).map(([k,v])=>featureRow(k,v)));
    list('trace',data.inference.trace.map(t=>t.rule+' · '+(t.passed?'PASS':'REVIEW')+' — '+t.explanation));
  }catch(error){
    $('error').textContent=error instanceof TypeError && /fetch/i.test(error.message)
      ? 'The CropDetect server is not reachable. Start it with start.bat, refresh this page, and try again.'
      : error.message;
    $('error').hidden=false
  }
  finally{$('analyze').disabled=false;$('analyze').innerHTML='Analyze leaf <span>↗</span>'}
});
async function init(){
  try{const h=await(await fetch('/api/health')).json();$('health').textContent=h.model_ready?'● Model ready · local inference':'Model training required'}catch{$('health').textContent='Server unavailable'}
  try{const samples=await(await fetch('/api/samples')).json();samples.forEach((sample,i)=>{
    const button=document.createElement('button');button.type='button';const img=document.createElement('img');img.src='/sample/'+i;img.alt=sample.actual.replaceAll('_',' ');
    const span=document.createElement('span');span.textContent=img.alt;button.append(img,span);
    button.onclick=async()=>{try{const r=await fetch('/sample/'+i);if(!r.ok)throw new Error('Sample unavailable');choose(new File([await r.blob()],sample.actual+'.jpg',{type:'image/jpeg'}))}catch(e){$('error').textContent=e.message;$('error').hidden=false}};$('samples').append(button);
  })}catch{}
  try{
    const response=await fetch('/api/metrics');if(!response.ok)throw new Error('Training results are not available yet.');const m=await response.json();
    for(const [name,value] of [['Test accuracy',m.accuracy],['Macro precision',m.report['macro avg'].precision],['Macro recall',m.report['macro avg'].recall],['Macro F1-score',m.report['macro avg']['f1-score']]]){
      const el=document.createElement('div');el.className='metric';const strong=document.createElement('strong');strong.textContent=(value*100).toFixed(1)+'%';const label=document.createElement('span');label.textContent=name;el.append(strong,label);$('metrics').append(el);
    }
    const description=document.createElement('p');description.className='scope';description.textContent=m.downloaded+' downloaded images · '+m.duplicates_removed+' exact pixel duplicates removed · '+m.rejected_images.length+' rejected · seed '+m.seed+'. Selected model: '+m.selected_model+'.';$('dataset').append(description);
    const table=document.createElement('table');const head=document.createElement('tr');for(const x of ['Class','Train','Val','Test']){const th=document.createElement('th');th.textContent=x;head.append(th)}table.append(head);
    for(const label of m.labels){const row=document.createElement('tr');for(const x of [label.replaceAll('_',' '),...['train','validation','test'].map(s=>m.split_counts[s][label]||0)]){const td=document.createElement('td');td.textContent=x;row.append(td)}table.append(row)}$('dataset').append(table);
    const pest=document.createElement('p');pest.className='scope';pest.textContent='Pest detector test accuracy: '+(m.pest_detector.accuracy*100).toFixed(1)+'%. Counts above are original images; training adds one augmented variant per accepted image.';$('dataset').append(pest);
    const a=document.createElement('a');a.href='https://github.com/spMohanty/PlantVillage-Dataset';a.textContent='PlantVillage source repository ↗';a.target='_blank';$('dataset').append(a);
  }catch(e){$('metrics').textContent=e.message}
}
init();
