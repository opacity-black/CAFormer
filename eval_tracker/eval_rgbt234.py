import rgbt
from seqList import where_seq_already

rgbt234 = rgbt.RGBT234()
result_path = "./tracking_result//caformer/vit_cte_cma_st_f3_lr5e5_ep015/rgbt234"   # 88.5, 66.1
result_path = "./tracking_result//caformer/vit_cte_cma_st_f3_lr5e5_ep025/rgbt234"   # 87.0, 64.9
seq_li,l = where_seq_already(result_path)
print(l)


rgbt234(
    tracker_name="T1",
    result_path=result_path,
    seqs=seq_li
)

result_path = "./tracking_result//caformer/vit_cte_cma_st_f3_lr5e5_ep015/rgbt234"   # 88.5, 66.1
rgbt234(
    tracker_name="T2",
    result_path=result_path,
    seqs=seq_li
)

if __name__=="__main__":

    mpr_dict = rgbt234.MPR(seqs=seq_li)

    print('')
    for k,v in mpr_dict.items():
        print(k, "MPR", round(v[0]*100, 1))

    msr_dict = rgbt234.MSR(seqs=seq_li)

    print('')
    for k,v in msr_dict.items():
        print(k, "MSR", round(v[0]*100, 1))
