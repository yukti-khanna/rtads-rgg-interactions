#!/usr/bin/env python3
"""Enumerate separate campaigns; mark exclusions before any selection."""
from control_campaign_common import *

def classify_existing(args, excluded, tp, bp, rp):
    lookup = {}
    path = args.input_dir/'simulation_sequence_lookup_400.csv'
    if path.exists():
        table = pd.read_csv(path,usecols=['variant_id','molecule_type','parent_id','biological_class','variant_type'])
        lookup = {r['variant_id']:r for r in table.to_dict('records')}
    # New metadata provides mappings for all additional eligible catalogue variants.
    for molecule,table,parent_col in [('TAD',tp,'parent_id'),('TAD',bp,'parent_id'),('RGG',rp,'RGG_ID')]:
        for row in table.to_dict('records'):
            for number in [None]+list(range(1,21)):
                if molecule=='TAD' and row['class']=='basic_AD_control' and number is not None:
                    continue  # Historical basic shuffles mapped from lookup only; never generate them.
                vid=variant(molecule,row[parent_col],number)
                lookup.setdefault(vid,dict(variant_id=vid,molecule_type=molecule,parent_id=row[parent_col],
                    biological_class=row['class'] if molecule=='TAD' else 'conventional_RGG',
                    variant_type='WT' if number is None else 'shuffle'))
    rows=[]
    for name in sorted(excluded):
        category='OTHER_EXISTING'; notes='unmapped or outside allowed new campaigns'
        r,t=name.split('__') if name.count('__')==1 else ('','')
        tm,rm=lookup.get(t),lookup.get(r)
        if tm and rm and tm['molecule_type']=='TAD' and rm['molecule_type']=='RGG':
            ts=tm['variant_type']!='WT'; rs=rm['variant_type']!='WT'
            if tm['biological_class']=='basic_AD_control':
                if ts or rs: category='EXISTING_UNWANTED_BASIC_SHUFFLE'; notes='forbidden in both new campaigns'
                elif rm['parent_id'] in set(rp.RGG_ID): category='EXISTING_BASIC_WT'; notes='WT basic + WT conventional RGG'
            elif tm['parent_id'] in set(tp.parent_id) and rm['parent_id'] in set(rp.RGG_ID) and ts!=rs:
                category='EXISTING_SHUFFLE'; notes='exactly one acidic/native side shuffled'
            elif not ts and not rs: notes='WT/WT reference; not counted as a new shuffle system'
        rows.append(dict(pair_name=name,category=category,TAD_parent=tm['parent_id'] if tm else '',
            TAD_variant=t,RGG_parent=rm['parent_id'] if rm else '',RGG_variant=r,notes=notes))
    audit=pd.DataFrame(rows)
    if len(audit)!=len(excluded) or audit.pair_name.nunique()!=len(excluded): fail('existing-pair classification accounting failed')
    audit.to_csv(args.output_dir/'existing_400_pair_classification.csv',index=False)
    files={'EXISTING_BASIC_WT':'existing_basic_WT_pairs.txt','EXISTING_SHUFFLE':'existing_shuffle_pairs.txt',
           'EXISTING_UNWANTED_BASIC_SHUFFLE':'existing_unwanted_basic_shuffle_pairs.txt','OTHER_EXISTING':'existing_other_pairs.txt'}
    for category,filename in files.items():
        (args.output_dir/filename).write_text(''.join(n+'\n' for n in audit[audit.category==category].pair_name))
    return audit.category.value_counts().to_dict()

def main():
    args = args_parser(__doc__).parse_args()
    excluded, digest = exclusion(args.existing_pairs), sha(args.existing_pairs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    native = read_table(args.input_dir/'native_RTAD_eligible_pool.csv', ['parent_id','gene','sequence'])
    native['class']='native_RTAD'
    ext = read_table(args.input_dir/'external_acidic_AD_parents.csv', ['parent_id','gene','class','sequence'])
    tp = pd.concat([native,ext],ignore_index=True)
    bp = read_table(args.input_dir/'basic_control_TAD_parents.csv', ['parent_id','gene','class','sequence'])
    rp = read_table(args.input_dir/'eligible_conventional_RGGs.csv', ['RGG_ID','UniProt_ID','sequence'])
    ts = read_table(args.input_dir/'shuffle_TAD_variants.csv', ['parent_id','shuffle_number','seed','sequence'])
    rs = read_table(args.input_dir/'shuffle_RGG_variants.csv', ['RGG_parent_id','shuffle_number','seed','sequence'])
    if len(native)<13 or len(ext)!=3 or set(ext.gene)!={'ETV4','SPI1','CREB3'}:
        fail('need entire native eligible pool and three fixed external ADs')
    if set(bp['gene']) != {'TCF7L2','TET3','EOMES','LEF1'} or len(bp) != 4:
        fail('basic parent set must contain exactly four predefined controls')
    for table,col,parents in [(ts,'parent_id',tp.parent_id),(rs,'RGG_parent_id',rp.RGG_ID)]:
        if table.groupby(col).size().to_dict() != {x:20 for x in parents}:
            fail('every shuffle parent must have 20 variants')
        if table.duplicated([col,'shuffle_number']).any() or table.duplicated([col,'sequence']).any():
            fail('duplicate shuffle within parent')
    tg = {p:g.to_dict('records') for p,g in ts.groupby('parent_id')}
    rg = {p:g.to_dict('records') for p,g in rs.groupby('RGG_parent_id')}
    configure_naming(args.input_dir)
    category_counts=classify_existing(args,excluded,tp,bp,rp)
    shuffle,refs,basic = [],[],[]
    for tad in tp.to_dict('records'):
        for rgg in rp.to_dict('records'):
            refs.append(candidate(tad,rgg,'WT_WT',excluded))
            shuffle.extend(candidate(tad,rgg,'TAD_SHUFFLE',excluded,tad_sh=s) for s in tg[tad['parent_id']])
            shuffle.extend(candidate(tad,rgg,'RGG_SHUFFLE',excluded,rgg_sh=s) for s in rg[rgg['RGG_ID']])
    for tad in bp.to_dict('records'):
        for rgg in rp.to_dict('records'):
            basic.append(candidate(tad,rgg,'WT_WT',excluded))
    sd,rd,bd = pd.DataFrame(shuffle),pd.DataFrame(refs),pd.DataFrame(basic)
    validate_shuffles(sd,tp,rp)
    for df in (sd,rd,bd):
        if df.pair_name.duplicated().any(): fail('duplicate candidate pair ID')
    sd.to_parquet(out/'shuffle_candidate_universe.parquet',index=False,compression='zstd')
    rd.to_csv(out/'shuffle_WT_reference_universe.csv',index=False)
    bd.to_csv(out/'basic_control_candidate_universe.csv',index=False)
    audits=[]
    for campaign,df in [('shuffle',sd),('shuffle_reference',rd),('basic_control',bd)]:
        removed=df[df.already_run_or_running].copy()
        removed['campaign']=campaign
        removed['exclusion_reason']='present_in_simulation_pair_names_400.txt'
        audits.append(removed[['pair_name','campaign','condition','TAD_parent','RGG_parent','exclusion_reason']])
    pd.concat(audits).to_csv(out/'excluded_existing_pairs_from_candidates.csv',index=False)
    all_names=set(sd.pair_name)|set(rd.pair_name)|set(bd.pair_name)
    unmatched=sorted(excluded-all_names)
    (out/'existing_pairs_not_mapped_to_new_candidate_universe.txt').write_text(''.join(x+'\n' for x in unmatched))
    (out/'already_used_pair_names_normalized.txt').write_text(''.join(x+'\n' for x in sorted(excluded)))
    original=args.existing_pairs.read_text().splitlines()
    meta=dict(exclusion_filename=str(args.existing_pairs.resolve()),exclusion_sha256=digest,number_loaded=len(excluded),
              shuffle_candidates_removed=int(sd.already_run_or_running.sum()),
              basic_control_candidates_removed=int(bd.already_run_or_running.sum()),unmatched=len(unmatched),
              original_line_count=len(original),non_empty_count=sum(bool(x.strip()) for x in original),
              existing_pair_category_counts=category_counts)
    (out/'candidate_exclusion_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if sha(args.existing_pairs) != digest: fail('exclusion input modified')
    print(f'Candidates: shuffle={len(sd)}, WT references={len(rd)}, basic={len(bd)}; exclusions={meta}')

if __name__ == '__main__': main()
