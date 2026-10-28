"""Apply the supplied experimental columns; preserve notes and frozen scores."""
import csv
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
rows=list(csv.DictReader((HERE/'fig6_parent_master.before.csv').open()))
fields=list(rows[0])
byid={r['pair_id']:r for r in rows}
changes=[]
pending=[]

def setvalue(row,field,value,source):
    value=str(value)
    if field not in fields:fields.append(field)
    old=row.get(field,'')
    if old!=value:changes.append(dict(pair_id=row['pair_id'],field=field,old=old,new=value,source=source))
    row[field]=value

for incoming in list(csv.reader((HERE/'experimental_update.tsv').open(),delimiter='\t'))[1:]:
    if len(incoming)<10 or incoming[2].strip()=='[RRGG]7':continue
    gene=incoming[1].split()[0].lower()
    shuffle='shuffle' in incoming[1].lower()
    partner=incoming[2].strip().replace('–','-')
    matches=[r for r in rows if r['AHD_gene'].lower()==gene
             and r['AHD_variant'].startswith('shuffle')==shuffle and r['RGG_partner']==partner]
    if len(matches)!=1:raise ValueError(('Unresolved identity',incoming[1:3],len(matches)))
    r=matches[0];outcome=incoming[4].strip();kd=incoming[8].strip();err=incoming[9].strip()
    if kd and outcome!='Kd_fit':
        pending.append(dict(pair_id=r['pair_id'],reason='Supplied outcome contradicts numeric Kd; unchanged pending confirmation'))
        continue
    if outcome not in ['Kd_fit','LB','NB']:raise ValueError(outcome)
    for field,value in [('Outcome',outcome),('Kd_uM',kd),('Kd_uM_err',err)]:setvalue(r,field,value,'pasted experimental columns')
    setvalue(r,'experiment_status',{'Kd_fit':'fitted Kd supplied','NB':'explicit NB','LB':'LB without fitted Kd'}[outcome],'curated outcome')

# Explicit follow-up instruction: use the new Kd and bound fraction for this pair.
for field,value in [('Kd_uM','708.26'),('Kd_uM_err','99.23')]:
    setvalue(byid['g3bp1_rgg__foxo4_cr3'],field,value,'user confirmed revised FOXO4–G3BP1 fit')

for bound in csv.DictReader((HERE/'bound_fractions.csv').open()):
    if bound['mapping_status']!='confirmed':
        pending.append(dict(pair_id=bound['pair_id'],reason=bound['mapping_status']))
        continue
    r=byid[bound['pair_id']]
    for source,target in [('fraction_bound','fraction_bound_highest_ligand'),('percent_bound','percent_bound_highest_ligand'),
                          ('fraction_bound_uncertainty','fraction_bound_highest_ligand_uncertainty'),('percent_bound_uncertainty','percent_bound_highest_ligand_uncertainty')]:
        setvalue(r,target,bound[source],'separate user-supplied bound-fraction table')
    setvalue(r,'highest_ligand_equivalents',format(float(bound['L_total'])/float(bound['P_total']),'g'),'L_total / P_total from supplied table')

for r in rows:
    if r['Kd_uM']:
        kd=float(r['Kd_uM']);err=r['Kd_uM_err']
        setvalue(r,'pKd',format(6-math.log10(kd),'.15g'),'recalculated from Kd in micromolar')
        setvalue(r,'pKd_error',format(float(err)/(kd*math.log(10)),'.15g') if err else '', 'first-order propagation of supplied Kd error')
    else:
        setvalue(r,'pKd','','no fitted Kd');setvalue(r,'pKd_error','','no fitted Kd')

with (ROOT/'fig6_parent_master.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
for name,data,columns in [('changes.csv',changes,['pair_id','field','old','new','source']),('pending_confirmation.csv',pending,['pair_id','reason'])]:
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(data)
print(f'Updated {len(set(x["pair_id"] for x in changes))} rows; {len(pending)} pending decisions.')
