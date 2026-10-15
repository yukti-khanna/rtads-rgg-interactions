#!/usr/bin/env python3
"""Select a separate balanced NEW basic-control campaign after exclusions."""
from control_campaign_common import *

def main():
    p=args_parser(__doc__)
    p.add_argument('--n-systems',type=int,default=200)
    args=p.parse_args()
    if args.n_systems<=0 or args.n_systems%4: fail('--n-systems must be positive and divisible by 4')
    excluded=exclusion(args.existing_pairs)
    meta=verify_source(args,excluded)
    df=read_table(args.input_dir/'basic_control_candidate_universe.csv',['pair_name','already_run_or_running','RGG_parent','RGG_UniProt','TAD_parent','TAD_gene'])
    if not (df.already_run_or_running==df.pair_name.isin(excluded)).all(): fail('exclusion flags inconsistent')
    available=df[~df.already_run_or_running].copy()  # Exclude BEFORE RNG.
    counts=available.groupby('RGG_parent').TAD_parent.nunique()
    valid=set(counts[counts==4].index)
    available=available[available.RGG_parent.isin(valid)]
    proteins=sorted(available.RGG_UniProt.unique())
    n=args.n_systems//4
    if len(proteins)<n: fail(f'need {n} distinct conventional proteins with all four pairs new; only {len(proteins)} eligible')
    ss=seed_for('new_basic_protein_selection',str(args.n_systems),args.seed)
    selected=np.array(proteins)[np.random.default_rng(ss).choice(len(proteins),n,replace=False)].tolist()
    rows=[]
    for order,uid in enumerate(selected,1):
        parents=sorted(available[available.RGG_UniProt==uid].RGG_parent.unique())
        rs=seed_for('new_basic_region_selection',f'{args.n_systems}|{uid}',args.seed)
        parent=parents[int(np.random.default_rng(rs).integers(len(parents)))]
        rows.append(dict(RGG_parent=parent,RGG_UniProt=uid,random_selection_order=order,
                         protein_selection_seed=ss,region_selection_seed=rs,all_four_pairs_confirmed_new=True))
    parents=[r['RGG_parent'] for r in rows]
    final=available[available.RGG_parent.isin(parents)].copy().sort_values(['TAD_parent','RGG_parent'])
    final['campaign']='basic_control'
    final['selection_seed']=ss
    if set(final.TAD_gene)!={'TCF7L2','TET3','EOMES','LEF1'} or final.TAD_parent.nunique()!=4: fail('basic control identity validation failed')
    if final.groupby('TAD_parent').size().ne(n).any() or final.RGG_parent.nunique()!=n or final.RGG_UniProt.nunique()!=n: fail('common context balance failed')
    if not (final.TAD_variant_type=='WT').all() or not (final.RGG_variant_type=='WT').all(): fail('basic campaign cannot contain shuffles')
    paths=outputs(args,final,('basic_control',args.n_systems),args.n_systems,n+4,excluded)
    if sha(args.existing_pairs) != meta['exclusion_sha256']: fail('exclusion input changed during selection')
    pd.DataFrame(rows).to_csv(args.output_dir/f'basic_control_selected_RGGs_{args.n_systems}.csv',index=False)
    audit_update(args,f'basic_control_{args.n_systems}',dict(candidate_exclusions=meta,eligible_distinct_proteins=len(proteins),
        selected_RGGs=rows,rejected_random_draws=[],n_systems=args.n_systems),paths)
    summary=['basic controls = TCF7L2, TET3, EOMES, LEF1','basic controls shuffled = NO','shuffled RGGs used = NO',
             f'eligible common RGG regions after exclusion = {len(valid)}',f'eligible common RGG proteins after exclusion = {len(proteins)}',
             f'requested NEW systems = {args.n_systems}',f'selected common RGG parents = {n}',
             f'total NEW basic-control simulations = {len(final)}','overlap with exclusion file = 0']
    (args.output_dir/f'SCRIPT4_NEW_BASIC_{args.n_systems}_REPORT.md').write_text('# Current final NEW basic-control campaign\n\n'+'\n'.join(summary)+'\n\n'+STATEMENT+'\n\nThe basic-control campaign contained only WT basic AD-derived sequences paired with WT conventional RGG regions.\n\nNo simulations were launched.\n')
    print('\n'.join(summary))
    for label,path in zip(['pair-list path','FASTA path','manifest path'],paths): print(f'{label}: {path}')

if __name__=='__main__': main()
