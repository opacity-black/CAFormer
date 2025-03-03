import rgbt
from seqList import where_seq_already

rgbt234 = rgbt.RGBT234()
result_path = ""
seq_li,l = where_seq_already(result_path)
print(l)


rgbt234(
    tracker_name="T1",
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
