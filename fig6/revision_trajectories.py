"""Stream matched WT trajectories using serialized force parameters; no simulations."""
from pathlib import Path
import hashlib,json,re,collections,sys,warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDReader
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.util import convert_aa_code
from numba import njit
warnings.filterwarnings('ignore',category=DeprecationWarning,module='MDAnalysis')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'revision_analysis/trajectories'
RUNROOT=Path('/Users/yuktikhanna/calvados/fig6/run_validated_final')
RUNROOTS=[RUNROOT]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sequence(a):return ''.join(convert_aa_code(n) for n in a.resnames)

def find_inputs(row):
 pdbs=[]
 for root in RUNROOTS:
  folder=root/('test_'+row.pair_id)/'298k_20.0nm'
  pdbs.extend(sorted(folder.glob('job_*/top.pdb')))
 good=[]
 for p in pdbs:
  u=mda.Universe(str(p));segments=list(u.segments)
  if len(segments)!=2:continue
  seq=[sequence(s.atoms) for s in segments]
  if seq==[row.RGG_sequence,row.AHD_sequence] or seq==[row.AHD_sequence,row.RGG_sequence]:
   xml=p.parent/(row.pair_id+'_20.0nm_'+p.parent.name+'.xml')
   if xml.exists():good.append((p,xml,seq))
 if not good:raise ValueError('No exact-sequence topology plus serialized System in inspected validated-final run')
 return good

def inventory(d):
 OUT.mkdir(parents=True,exist_ok=True);rows=[];partners=set(d[d.AHD_name.eq('lef1_nter')].RGG_name)
 for _,r in d[~d.is_shuffle & d.RGG_name.isin(partners)].iterrows():
  f=ROOT/'validated_dcds'/('combined_trajectory_'+r.pair_id+'_20nm_sliced.dcd')
  a=dict(pair_id=r.pair_id,dcd=str(f),status='missing DCD',reason='',AHD_name=r.AHD_name,RGG_name=r.RGG_name)
  if f.exists():
   try:
    with DCDReader(str(f)) as reader:
     a.update(frames=len(reader),atoms=reader.n_atoms,box_A=';'.join(map(str,reader[0].dimensions)),dcd_bytes=f.stat().st_size)
    good=find_inputs(r);p,x,seq=good[0]
    if a['atoms']!=len(seq[0])+len(seq[1]):raise ValueError('DCD/topology atom count mismatch')
    a.update(status='ready for coordinate linkage check',topology=str(p),system_xml=str(x),topology_sha256=sha(p),system_sha256=sha(x),sequence_check='both full sequences match registry',available_replica_topologies=len(good))
   except Exception as e:a.update(status='excluded unresolved mapping',reason=str(e))
  rows.append(a)
 pd.DataFrame(rows).to_csv(OUT/'inventory.csv',index=False)
 return pd.DataFrame(rows)

def parameters(xml,n):
 forces=ET.parse(xml).getroot().findall('./Forces/Force');nb=[f for f in forces if f.get('type')=='CustomNonbondedForce'];by={f.get('forceGroup'):f for f in nb}
 ah=by['0'];yu=by['1'];expr=ah.get('energy');yexpr=yu.get('energy')
 if not expr.startswith('0.8368') or 'l=select(id1+id2' not in expr:raise ValueError('Unsupported AH expression; inspect rather than guess')
 if ah.get('method')!='2' or yu.get('method')!='2':raise ValueError('Expected periodic cutoff')
 s=np.array([[float(p.get('param'+str(k))) for k in [1,2,3]] for p in ah.findall('./Particles/Particle')]);q=np.array([float(p.get('param1')) for p in yu.findall('./Particles/Particle')])
 match=re.fullmatch(r'q\*([0-9.eE+-]+)\*\(exp\(-([0-9.eE+-]+)\*r\)/r-([0-9.eE+-]+)\); q=q1\*q2',yexpr)
 if not match or len(s)!=n or len(q)!=n:raise ValueError('Unsupported electrostatic expression/particle count')
 eps=float(expr.split('*select')[0]);ey,kappa,shift=map(float,match.groups())
 return s,q,np.array([eps,float(ah.get('cutoff')),ey,kappa,shift,float(yu.get('cutoff'))]),[expr,yexpr]

@njit
def accumulate(dist,sig,lam,qq,par,sum_ah,sum_yu,sum_contact,sum_cond,cls,clsum):
 ea=0.;ey=0.;contact=False;cutcounts=np.zeros(3);atr=0.;rep=0.
 for i in range(dist.shape[0]):
  for j in range(dist.shape[1]):
   r=dist[i,j]*.1
   if r<=0:raise ValueError('Zero interchain distance')
   if r<.6:cutcounts[0]+=1
   if r<.8:cutcounts[1]+=1;contact=True;sum_contact[i,j]+=1
   if r<1.:cutcounts[2]+=1
   a=0.;y=0.
   if r<=par[1]:
    sr6=(sig[i,j]/r)**6;shift=(sig[i,j]/par[1])**12-(sig[i,j]/par[1])**6
    if r<=2**(1/6)*sig[i,j]:a=par[0]*(4*(sr6*sr6-sr6-lam[i,j]*shift)+1-lam[i,j])
    else:a=par[0]*4*lam[i,j]*(sr6*sr6-sr6-shift)
   if r<=par[5]:y=par[2]*qq[i,j]*(np.exp(-par[3]*r)/r-par[4])
   sum_ah[i,j]+=a;sum_yu[i,j]+=y;ea+=a;ey+=y
   total=a+y;c=cls[i,j]
   clsum[c,0]+=a;clsum[c,1]+=y;clsum[c,2]+=min(total,0);clsum[c,3]+=max(total,0)
   atr+=min(total,0);rep+=max(total,0)
 # Conditional energy maps require a second traversal only for encountered frames.
 if contact:
  for i in range(dist.shape[0]):
   for j in range(dist.shape[1]):
    r=dist[i,j]*.1;a=0.;y=0.
    if r<=par[1]:
     sr6=(sig[i,j]/r)**6;shift=(sig[i,j]/par[1])**12-(sig[i,j]/par[1])**6
     if r<=2**(1/6)*sig[i,j]:a=par[0]*(4*(sr6*sr6-sr6-lam[i,j]*shift)+1-lam[i,j])
     else:a=par[0]*4*lam[i,j]*(sr6*sr6-sr6-shift)
    if r<=par[5]:y=par[2]*qq[i,j]*(np.exp(-par[3]*r)/r-par[4])
    sum_cond[i,j]+=a+y
 has_contact=1.0 if contact else 0.0
 has_06=1.0 if cutcounts[0]>0 else 0.0
 has_10=1.0 if cutcounts[2]>0 else 0.0
 return np.array([ea,ey,ea+ey,has_contact,has_06,has_10,cutcounts[1],atr,rep])

def residue_class(s):return 0 if s in 'DE' else 1 if s in 'KRH' else 2 if s in 'FWY' else 3

def analyze(row,meta):
 f=Path(meta.dcd);p=Path(meta.topology);xml=Path(meta.system_xml);dest=OUT/'cache'/row.pair_id;dest.mkdir(parents=True,exist_ok=True)
 good=find_inputs(row)
 archived=[]
 for top,x,_ in good:
  archived.extend([dict(path=str(top),sha256=sha(top)),dict(path=str(x),sha256=sha(x))])
  archived.extend(dict(path=str(src),sha256=sha(src)) for src in sorted(top.parent.glob('*.dcd')))
 key=dict(dcd_sha256=sha(f),topology_sha256=sha(p),system_sha256=sha(xml),code_sha256=sha(__file__),cutoffs_nm=[.6,.8,1.],frame_policy='all supplied sliced frames; contiguous-block uncertainty when replica boundaries are unavailable')
 key['archived_sources']=archived
 if (dest/'provenance.json').exists() and json.loads((dest/'provenance.json').read_text()).get('signature')==key and (dest/'summary.csv').exists():return pd.read_csv(dest/'summary.csv').iloc[0].to_dict()
 u=mda.Universe(str(p));ss=list(u.segments);rggidx=0 if sequence(ss[0].atoms)==row.RGG_sequence else 1;rg=ss[rggidx].atoms.indices;aa=ss[1-rggidx].atoms.indices
 s,q,par,expr=parameters(xml,len(u.atoms))
 sig=(s[rg,0,None]+s[aa,0][None,:])/2;lam=(s[rg,1,None]+s[aa,1][None,:])/2*s[rg,2,None]*s[aa,2][None,:];qq=q[rg,None]*q[aa][None,:]
 # Do not allow unaccounted interchain exclusions.
 for force in ET.parse(xml).getroot().findall('./Forces/Force'):
  if force.get('type')=='CustomNonbondedForce':
   for ex in force.findall('./Exclusions/Exclusion'):
    i=int(ex.get('p1'));j=int(ex.get('p2'))
    if (i in rg and j in aa) or (j in rg and i in aa):raise ValueError('Interchain force exclusion requires explicit handling')
 # Verify every available job against the same force parameters. The supplied
 # combined DCDs contain transformed coordinates, so their frames cannot be
 # assigned reliably to raw jobs by byte-level coordinate hashes.
 sourcefiles=[]
 for top,x,_ in good:
  _,_,pp,ee=parameters(x,len(u.atoms));sx,qx,_,_=parameters(x,len(u.atoms))
  if not (np.array_equal(s,sx) and np.array_equal(q,qx) and np.array_equal(par,pp) and expr==ee):raise ValueError('Replica force parameters disagree')
  for src in sorted(top.parent.glob('*.dcd')):
   sourcefiles.append(next(z for z in archived if z['path']==str(src)))
 a=np.zeros(sig.shape);y=a.copy();ct=a.copy();cond=a.copy();classes=np.array([[residue_class(i)*4+residue_class(j) for j in row.AHD_sequence] for i in row.RGG_sequence]);clsum=np.zeros((16,4));frame=[];verify=[]
 # Use Figure 4 potentials as an independent implementation with exact System parameters.
 sys.path.insert(0,str(ROOT.parent/'fig4'));from energy_funcs import ah_potential,yukawa_potential
 with DCDReader(str(f)) as reader:
  for k,ts in enumerate(reader):
   if ts.dimensions is None or not np.allclose(ts.dimensions[3:],90):raise ValueError('Absent/nonorthogonal box requires explicit verification')
   dm=distance_array(ts.positions[rg],ts.positions[aa],box=ts.dimensions)
   v=accumulate(dm,sig,lam,qq,par,a,y,ct,cond,classes,clsum);frame.append(v)
   if k in [0,len(reader)//2,len(reader)-1]:
    ar=yr=0.
    for i in range(len(rg)):
     for j in range(len(aa)):
      ar+=ah_potential(dm[i,j]/10,sig[i,j],par[0],lam[i,j],par[1]);yr+=yukawa_potential(dm[i,j]/10,qq[i,j],par[3],par[2],par[5])
    if not np.allclose([ar,yr],v[:2],atol=1e-7,rtol=1e-8):raise ValueError('Figure 4 energy reference mismatch')
    verify.append(dict(frame=k,AH_reference=ar,AH_stream=v[0],Yukawa_reference=yr,Yukawa_stream=v[1],passed=True))
 values=np.asarray(frame);n=len(values);nc=int(values[:,3].sum());names=['AH_kJ_mol','Yukawa_kJ_mol','total_kJ_mol','contact_08','contact_06','contact_10','contacts_08','attractive_total_kJ_mol','repulsive_total_kJ_mol'];df=pd.DataFrame(values,columns=names);df['replica']='unresolved_combined_trajectory'
 # Full frame rows are retained in compressed CSV; no independent-frame uncertainty.
 df.to_csv(dest/'frame_summary.csv.gz',index=False);rep=df.groupby('replica')[names].mean();rep.to_csv(dest/'replica_means.csv')
 blocks=[]
 for b,idx in enumerate(np.array_split(np.arange(n),10)):
  if len(idx):blocks.append(dict(block=b,first_frame=int(idx[0]),last_frame=int(idx[-1]),**df.iloc[idx][names].mean().to_dict()))
 blockdf=pd.DataFrame(blocks);blockdf.to_csv(dest/'block_sensitivity.csv',index=False)
 summary=dict(pair_id=row.pair_id,frames=n,n_replicas=0,contiguous_blocks=len(blockdf),contact_frames=nc,possible_residue_pairs=sig.size,uncertainty='SEM across 10 contiguous blocks; blocks are sensitivity units, not independent replicas',**df[names].mean().to_dict())
 for c in names:
  summary[c+'_replica_SEM']=np.nan
  summary[c+'_block_SEM']=blockdf[c].sem()
  summary[c+'_contact_conditioned']=df.loc[df.contact_08.eq(1),c].mean()
 for c in names[:3]:summary[c+'_per_possible_pair']=summary[c]/sig.size
 rr=[]
 for i,ri in enumerate(row.RGG_sequence):
  for j,aj in enumerate(row.AHD_sequence):rr.append(dict(RGG_index=i+1,AHD_index=j+1,RGG_residue=ri,AHD_residue=aj,AH_kJ_mol=a[i,j]/n,Yukawa_kJ_mol=y[i,j]/n,total_kJ_mol=(a[i,j]+y[i,j])/n,contact_occupancy=ct[i,j]/n,contact_conditioned_total=cond[i,j]/nc if nc else np.nan))
 pd.DataFrame(rr).to_csv(dest/'residue_maps.csv.gz',index=False)
 classnames=['Acidic','Basic','Aromatic','Other'];cr=[]
 for i in range(16):cr.append(dict(RGG_class=classnames[i//4],AHD_class=classnames[i%4],AH_kJ_mol=clsum[i,0]/n,Yukawa_kJ_mol=clsum[i,1]/n,attractive_total_kJ_mol=clsum[i,2]/n,repulsive_total_kJ_mol=clsum[i,3]/n))
 pd.DataFrame(cr).to_csv(dest/'residue_class_contributions.csv',index=False)
 if not np.allclose([a.sum()/n,y.sum()/n],df[names[:2]].mean(),rtol=1e-8,atol=1e-7):raise ValueError('Residue/pair aggregation mismatch')
 pd.DataFrame(verify).to_csv(dest/'energy_reference_verification.csv',index=False);pd.DataFrame([summary]).to_csv(dest/'summary.csv',index=False)
 (dest/'provenance.json').write_text(json.dumps(dict(signature=key,source_trajectories=sourcefiles,replica_coordinate_linkage='unresolved after coordinate-transformed combined DCD was supplied; no replica SEM reported',uncertainty='ten contiguous blocks of the supplied combined trajectory; sensitivity only, not independent replicas',expressions=expr,parameters=par.tolist(),sequence_orientation='rows RGG; columns AHD',units='distances nm; signed energies kJ/mol'),indent=2))
 return summary

def plot(d,summary):
 from run_revision_analysis import export,save
 import matplotlib.pyplot as plt
 if summary.empty:return
 z=summary.merge(d[['pair_id','AHD_name','RGG_name','Outcome']],on='pair_id',validate='one_to_one');save(z,'tables/trajectory_pair_summary.csv')
 lef=z[z.AHD_name.eq('lef1_nter')];rows=[]
 for _,r in lef.iterrows():
  for _,c in z[z.RGG_name.eq(r.RGG_name)&z.AHD_name.ne('lef1_nter')].iterrows():
   a=dict(LEF1_pair=r.pair_id,comparator_pair=c.pair_id,partner=r.RGG_name)
   for field in ['AH_kJ_mol','Yukawa_kJ_mol','total_kJ_mol','contact_08','total_kJ_mol_contact_conditioned','total_kJ_mol_per_possible_pair']:a['delta_'+field]=r[field]-c[field]
   rows.append(a)
 save(pd.DataFrame(rows),'tables/LEF1_matched_trajectory_deltas.csv')
 for field in ['AH_kJ_mol','Yukawa_kJ_mol','total_kJ_mol','contact_08','total_kJ_mol_contact_conditioned']:
  fig,ax=plt.subplots(figsize=(8,4));partners=sorted(z.RGG_name.unique())
  for k,p in enumerate(partners):
   g=z[z.RGG_name.eq(p)].sort_values('AHD_name')
   for j,(_,r) in enumerate(g.iterrows()):ax.scatter(k+(j-(len(g)-1)/2)*.08,r[field],color='#D55E00' if r.AHD_name=='lef1_nter' else '#999999',s=45 if r.AHD_name=='lef1_nter' else 20)
  ax.scatter([],[],c='#D55E00',label='LEF1');ax.scatter([],[],c='#999999',label='Matched WT AHDs');ax.legend(frameon=False);ax.set(xticks=range(len(partners)),xticklabels=partners,ylabel=field);ax.tick_params(axis='x',rotation=30)
  export(fig,'trajectory_'+field,z,'All supplied combined-trajectory frames; exact sequences and serialized System parameters verified. LEF1 highlighted. Signed energy is a model-dependent ensemble observable, not a B22 decomposition. Uncertainty uses contiguous-block sensitivity because replica boundaries are unresolved.')
 for _,r in lef.iterrows():
  frame=pd.read_csv(OUT/'cache'/r.pair_id/'residue_maps.csv.gz');fig,axs=plt.subplots(1,3,figsize=(13,4))
  for ax,col in zip(axs,['contact_occupancy','AH_kJ_mol','Yukawa_kJ_mol']):
   arr=frame.pivot(index='RGG_index',columns='AHD_index',values=col);kw={}
   if col!='contact_occupancy':v=max(abs(arr.to_numpy()).max(),1e-8);kw=dict(vmin=-v,vmax=v)
   im=ax.imshow(arr,aspect='auto',cmap='viridis' if col=='contact_occupancy' else 'coolwarm',**kw);fig.colorbar(im,ax=ax,shrink=.8);ax.set(title=col,xlabel='LEF1 construct index',ylabel='RGG construct index')
  export(fig,'residue_map_'+r.pair_id,frame,'Construct-relative residue indices; energies are all-frame means in kJ/mol. Contact occupancy is fraction of frames below 0.8 nm. No experimental binding interface is inferred.')

def run(d):
 inv=inventory(d);summary=[]
 for i,r in inv.iterrows():
  if r.status not in ['ready for coordinate linkage check','completed']:continue
  print('Trajectory',r.pair_id,flush=True)
  try:
   result=analyze(d.set_index('pair_id',drop=False).loc[r.pair_id],r);summary.append(result);inv.loc[i,'status']='completed';inv.loc[i,'reason']='combined trajectory analyzed; exact sequences and force parameters verified; replica linkage unresolved, so contiguous-block sensitivity reported'
  except Exception as e:
   inv.loc[i,'status']='excluded unresolved provenance or failed validation';inv.loc[i,'reason']=str(e);print('Excluded',r.pair_id,str(e),flush=True)
  inv.to_csv(OUT/'inventory.csv',index=False)
 pd.DataFrame(summary,columns=list(summary[0]) if summary else ['pair_id','frames','n_replicas']).to_csv(OUT/'summary.csv',index=False)
 plot(d,pd.DataFrame(summary))
