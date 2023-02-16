class SearchSpace:
# Slice bounds
    s_h_lb = 3840
    s_h_ub = 3840
    s_h_res = 512
    s_h = list(range(s_h_lb, s_h_ub+s_h_res, s_h_res))

    s_w_lb = 3840
    s_w_ub = 3840
    s_w_res = 512
    s_w = list(range(s_w_lb, s_w_ub+s_w_res, s_w_res))

# Resize bounds of slice
    r_h_lb = 640
    r_h_ub = 1280
    r_h_res = 320
    r_h = list(range(r_h_lb, r_h_ub+r_h_res, r_h_res))

    r_w_lb = 640
    r_w_ub = 1280
    r_w_res = 320
    r_w = list(range(r_w_lb, r_w_ub+r_w_res, r_w_res)) 
ss = SearchSpace()
print(ss.s_h)
print(ss.s_w)
print(ss.r_h)
print(ss.r_w)
