q = tl.load(q_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)
k = tl.load(k_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)