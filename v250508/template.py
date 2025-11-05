# sbatch template1.sbatch; sbatch template2.sbatch; sbatch template3.sbatch; sbatch template4.sbatch; sbatch template5.sbatch; sbatch template6.sbatch; sbatch template7.sbatch; sbatch template8.sbatch

ics = [4,8,12,16,20,24,28,32]
#ics = [364,368,372,376,380,384,388,392]
#ics = [728,732,736,740,744,748,752,756]
#ics = [1096,1100,1104,1108,1112,1116,1120,1124]




for i,ic in enumerate(ics):
    fili = open("template.sh")
    filo = open("template"+str(i+1)+".sbatch",'w')
    for raw in fili:
        if "out_name=/work/choutilin1/out_vars33/vars33_V/" in raw:
            filo.write("out_name=/work/choutilin1/out_vars33/vars33_V/allMsst_"+str(ic).zfill(4)+".nc\n")
        elif "--samples_offset=XXXX" in raw:
            filo.write("        --samples_offset="+str(ic)+"\n")
        else:
            filo.write(raw)
    filo.close()
    fili.close()


print("sbatch template1.sbatch; sbatch template2.sbatch; sbatch template3.sbatch; sbatch template4.sbatch; sbatch template5.sbatch; sbatch template6.sbatch; sbatch template7.sbatch; sbatch template8.sbatch")


