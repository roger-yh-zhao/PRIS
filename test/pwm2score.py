
import numpy as np

def pwm2scores(pdb):
    seqs=open(f'{pdb}/seq.txt').readlines();seqs=[i[:-1] for i in seqs]
    
    with open(f'result/{pdb}pwm.dat') as f:
        text = f.read()


    blocks = [b.strip() for b in text.split("###") if b.strip()]


    arrays = [
    np.loadtxt(block.splitlines())
    for block in blocks
    ]


    
    scores=[]
    for array in arrays:
        for seq in seqs:        
            score = get_score( array, seq )
            scores.append( score )
  
    
    out = open(f'{pdb}/priseq.txt','w')
    for i in scores:
        out.writelines(f'{i}\n')
    
    return scores

def get_score( array, seq ):
    #pwmACGU
    score=0
    for i in range(len(seq)):
        s = seq[i]
        index = 'ACGU'.index(s)
        
        score += array[i][index]
 
    return score



pwm2scores('1zdk')
