import rgbt
from seqList import *
lasher = rgbt.LasHeR()

result_path = "./tracking_result//caformer/vit_cte_cma_st_f3_lr5e5_ep015/lashertestingset"
seq_list,l = where_seq_already(result_path)
print(l)

print("序列数", len(seq_list))


lasher(
    tracker_name="T1",
    result_path=result_path,
    seqs=seq_list
)


# lasher.draw_attributeRadar(metric_fun=lasher.PR, filename="eval_tracker/CAiATrack_lasher_PR.png")
# lasher.draw_attributeRadar(metric_fun=lasher.SR, filename="eval_tracker/CAiATrack_lasher_SR.png")

if __name__=="__main__":

    print('')
    pr_dict = lasher.PR(seqs=seq_list)
    for k,v in pr_dict.items():
        print("PR", k, round(v[0]*100, 1))

    print('')
    npr_dict = lasher.NPR(seqs=seq_list)
    for k,v in npr_dict.items():
        print("NPR", k, round(v[0]*100, 1))

    print('')
    sr_dict = lasher.SR(seqs=seq_list)
    for k,v in sr_dict.items():
        print("SR", k, round(v[0]*100, 1))

    print('')