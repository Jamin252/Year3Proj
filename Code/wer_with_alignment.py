import string
from typing import List, Dict

def wer_with_alignment(ref: List[str], hyp: List[str], normalize: bool = True, return_alignment: bool = True) -> Dict[str, object]:
    """
    Standard Levenshtein-based WER with alignment.
    """
    m, n = len(ref), len(hyp)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    ptr = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
        ptr[i][0] = 'del'
    for j in range(n + 1):
        d[0][j] = j
        ptr[0][j] = 'ins'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
                ptr[i][j] = 'match'
            else:
                sub = d[i-1][j-1] + 1
                ins = d[i][j-1] + 1
                dele = d[i-1][j] + 1
                
                d[i][j] = min(sub, ins, dele)
                if d[i][j] == sub:
                    ptr[i][j] = 'sub'
                elif d[i][j] == ins:
                    ptr[i][j] = 'ins'
                else:
                    ptr[i][j] = 'del'

    distance = d[m][n]
    wer = distance / m if m > 0 else (1.0 if n > 0 else 0.0)

    alignment = []
    if return_alignment:
        i, j = m, n
        while i > 0 or j > 0:
            op = ptr[i][j]
            if op == 'match':
                alignment.append(('match', 'REF', ref[i-1], hyp[j-1]))
                i -= 1
                j -= 1
            elif op == 'sub':
                alignment.append(('sub', 'REF', ref[i-1], hyp[j-1]))
                i -= 1
                j -= 1
            elif op == 'ins':
                alignment.append(('ins', None, None, hyp[j-1]))
                j -= 1
            elif op == 'del':
                alignment.append(('del', 'REF', ref[i-1], None))
                i -= 1
        alignment.reverse()

    return {
        'distance': distance,
        'wer': wer,
        'alignment': alignment
    }
