#!/usr/bin/env python3
"""Select exactly 400 NEW order-perturbation systems after exact exclusions."""
from control_campaign_common import *

def main():
    p=args_parser(__doc__)
    p.add_argument('--max-draws',type=int,default=10000)
    args=p.parse_args()
    excluded=exclusion(args.existing_pairs)
    meta=verify_source(args,excluded)
    df=read_table(args.input_dir/'shuffle_candidate_universe.parquet', ['pair_name','already_run_or_running','condition'])
    if not (df.already_run_or_running == df.pair_name.isin(excluded)).all(): fail('candidate exclusion flags inconsistent')
    eligible=df[~df.already_run_or_running].copy()  # BEFORE any RNG operation.
    native=read_table(args.input_dir/'native_RTAD_eligible_pool.csv',['parent_id','gene','UniProt_ID','sequence'])
    native['class']='native_RTAD'
    ext=read_table(args.input_dir/'external_acidic_AD_parents.csv',['parent_id','gene','class','sequence'])
    rp=read_table(args.input_dir/'eligible_conventional_RGGs.csv',['RGG_ID','UniProt_ID','sequence'])
    if len(native)<13 or native.parent_id.nunique()!=len(native) or native.gene.nunique()!=len(native) or native.UniProt_ID.nunique()!=len(native): fail('native pool must have at least 13 unique genes/proteins')
    if len(ext)!=3 or set(ext.gene)!={'ETV4','SPI1','CREB3'}: fail('incorrect external AD set')
    proteins=sorted(rp.UniProt_ID.unique())
    if len(proteins)<5: fail('fewer than five eligible proteins')
    rejected=[]
    draw_seed=seed_for('new_shuffle_common_protein_draws','five',args.seed)
    rng=np.random.default_rng(draw_seed)
    native_seed=seed_for('native_RTAD_selection','eligible_pool',args.seed)
    native_rng=np.random.default_rng(native_seed)
    native=native.sort_values('parent_id').reset_index(drop=True)
    for draw in range(1,args.max_draws+1):
        selected_native=native.iloc[native_rng.choice(len(native),13,replace=False)].copy()
        tp=pd.concat([selected_native,ext],ignore_index=True)
        chosen=np.array(proteins)[rng.choice(len(proteins),5,replace=False)].tolist()
        common=[]
        for order,uid in enumerate(chosen,1):
            regions=rp[rp.UniProt_ID==uid].sort_values('RGG_ID')
            rs=seed_for('new_shuffle_region_draw',f'{draw}|{uid}',args.seed)
            row=regions.iloc[int(np.random.default_rng(rs).integers(len(regions)))]
            common.append(dict(RGG_parent=row.RGG_ID,RGG_UniProt=uid,WT_sequence=row.sequence,
                               random_selection_order=order,protein_selection_seed=draw_seed,region_selection_seed=rs))
        parents=[x['RGG_parent'] for x in common]
        td=eligible[(eligible.condition=='TAD_SHUFFLE') & eligible.RGG_parent.isin(parents) & eligible.TAD_parent.isin(tp.parent_id)]
        tcounts=td.groupby(['TAD_parent','TAD_variant']).RGG_parent.nunique()
        valid_t=set(tcounts[tcounts==5].index)
        rd=eligible[(eligible.condition=='RGG_SHUFFLE') & eligible.RGG_parent.isin(parents) & eligible.TAD_parent.isin(tp.parent_id)]
        rcounts=rd.groupby(['RGG_parent','RGG_variant']).TAD_parent.nunique()
        valid_r=set(rcounts[rcounts==16].index)
        reasons=[]
        for parent in tp.parent_id:
            n=sum(x[0]==parent for x in valid_t)
            if n<4: reasons.append(f'{parent}: only {n} TAD shuffles new across all five RGGs; need 4')
        for parent in parents:
            if not any(x[0]==parent for x in valid_r): reasons.append(f'{parent}: no RGG shuffle new across all 16 WT TADs')
        if not reasons: break
        rejected.append(dict(draw=draw,proposed_native_RTADs=selected_native.parent_id.tolist(),native_selection_seed=native_seed,
                             proposed_common_RGGs=common,reasons=reasons))
    else:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'shuffle_rejected_draws.json').write_text(json.dumps(rejected,indent=2)+'\n')
        fail(f'no feasible common RGG set found in {args.max_draws} run-status-only draws; see shuffle_rejected_draws.json')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(common).to_csv(args.output_dir/'shuffle_selected_common_RGGs.csv',index=False)
    selected_native['selection_seed']=native_seed
    selected_native['random_selection_order']=range(1,14)
    selected_native.to_csv(args.output_dir/'selected_native_RTADs_for_new400.csv',index=False)
    tp.to_csv(args.output_dir/'shuffle_selected_TAD_parents.csv',index=False)
    selections=[]; selected_t=[]; selected_r=[]
    for parent in sorted(tp.parent_id):
        variants=sorted(v for p,v in valid_t if p==parent)
        ss=seed_for('new_four_TAD_shuffles',parent,args.seed)
        selected=np.array(variants)[np.random.default_rng(ss).choice(len(variants),4,replace=False)].tolist()
        for order,vid in enumerate(selected,1):
            rows=td[(td.TAD_parent==parent)&(td.TAD_variant==vid)].copy()
            rows['selection_seed']=ss
            selections.append(rows)
            row=rows.iloc[0]
            selected_t.append(dict(TAD_parent=parent,selected_shuffle_number=int(row.TAD_shuffle_number),
                variant_id=vid,shuffle_seed=int(row.shuffle_seed),selection_seed=ss,
                random_selection_order=order,all_five_pairs_confirmed_new=True))
    for parent in parents:
        variants=sorted(v for p,v in valid_r if p==parent)
        ss=seed_for('new_one_RGG_shuffle',parent,args.seed)
        vid=variants[int(np.random.default_rng(ss).integers(len(variants)))]
        rows=rd[(rd.RGG_parent==parent)&(rd.RGG_variant==vid)].copy()
        rows['selection_seed']=ss
        selections.append(rows)
        row=rows.iloc[0]
        selected_r.append(dict(RGG_parent=parent,RGG_UniProt=row.RGG_UniProt,
            selected_shuffle_number=int(row.RGG_shuffle_number),variant_id=vid,shuffle_seed=int(row.shuffle_seed),
            selection_seed=ss,all_sixteen_pairs_confirmed_new=True))
    pd.DataFrame(selected_t).to_csv(args.output_dir/'selected_TAD_shuffles_for_new400.csv',index=False)
    pd.DataFrame(selected_r).to_csv(args.output_dir/'selected_RGG_shuffles_for_new400.csv',index=False)
    final=pd.concat(selections,ignore_index=True)
    final['campaign']='shuffle'
    if final.condition.value_counts().to_dict()!={'TAD_SHUFFLE':320,'RGG_SHUFFLE':80}: fail('incorrect condition totals')
    if final[final.condition=='TAD_SHUFFLE'].groupby('TAD_variant').RGG_parent.nunique().ne(5).any(): fail('TAD shuffle reuse failed')
    if final[final.condition=='RGG_SHUFFLE'].groupby('RGG_variant').TAD_parent.nunique().ne(16).any(): fail('RGG shuffle reuse failed')
    validate_shuffles(final,tp,rp)
    paths=outputs(args,final,('shuffle',400),400,90,excluded)
    if sha(args.existing_pairs) != meta['exclusion_sha256']: fail('exclusion input changed during selection')
    refs=read_table(args.input_dir/'shuffle_WT_reference_universe.csv',['pair_name','TAD_parent','RGG_parent'])
    refs=refs[refs.RGG_parent.isin(parents) & refs.TAD_parent.isin(tp.parent_id)].copy()
    if len(refs)!=80: fail('WT reference status must have 80 rows')
    refs['already_run_or_running']=refs.pair_name.isin(excluded)
    refs['in_existing_400']=refs['already_run_or_running']
    refs['status']=np.where(refs.already_run_or_running,'existing_reference','missing_reference')
    refs=refs.rename(columns={'pair_name':'WT_pair_name'})
    refs[['WT_pair_name','TAD_parent','RGG_parent','in_existing_400','status']].to_csv(args.output_dir/'shuffle_WT_reference_status.csv',index=False)
    missing=refs[~refs.already_run_or_running].WT_pair_name.tolist()
    (args.output_dir/'shuffle_missing_WT_reference_pairs.txt').write_text(''.join(x+'\n' for x in missing))
    audit_update(args,'shuffle',dict(candidate_exclusions=meta,selected_native_RTADs=tp[tp['class']=='native_RTAD'][['parent_id','gene']].to_dict('records'),
        external_acidic_controls=tp[tp['class']=='external_AD_not_RTAD'][['parent_id','gene']].to_dict('records'),
        native_selection_seed=native_seed,
        common_RGGs=common,TAD_shuffles=selected_t,RGG_shuffles=selected_r,rejected_random_draws=rejected,
        successful_draw=draw,WT_references_existing=80-len(missing),WT_references_missing=len(missing)),paths)
    summary=[f'native R-TAD eligible pool = {len(native)}','shuffle TAD parents = 16','native R-TAD parents = 13','external acidic parents = ETV4, SPI1, CREB3',
        'common RGG parents = 5','selected TAD shuffle variants = 64','selected RGG shuffle variants = 5',
        'NEW TAD-shuffle simulations = 320','NEW RGG-shuffle simulations = 80','total NEW shuffle simulations = 400',
        'overlap with exclusion file = 0',f'WT references already available = {80-len(missing)}',f'WT references missing = {len(missing)}']
    (args.output_dir/'SCRIPT3_NEW_SHUFFLE_400_REPORT.md').write_text('# Current final NEW shuffle campaign\n\n'+'\n'.join(summary)+'\n\n'+STATEMENT+'\n\nBasic AD-derived controls were excluded entirely from the shuffle campaign. Basic-control sequences were not shuffled and were not paired with shuffled RGG sequences.\n\nMissing WT references are not included in the 400 systems. No simulations were launched.\n')
    print('\n'.join(summary))
    for label,path in zip(['pair-list path','FASTA path','manifest path'],paths): print(f'{label}: {path}')

if __name__=='__main__': main()
