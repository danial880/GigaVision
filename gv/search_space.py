class SearchSpace:
# Slice bounds
    # slice height lower bounds
    s_h_lb = 3840
    # slice height upper bounds
    s_h_ub = 3840
    # slice height step
    s_h_res = 512
    # list of slice heights 
    s_h = list(range(s_h_lb, s_h_ub+s_h_res, s_h_res))

    # slice width lower bounds
    s_w_lb = 3840
    # slice width upper bounds
    s_w_ub = 4352
    # slice width step
    s_w_res = 512
    # list of slice widths
    s_w = list(range(s_w_lb, s_w_ub+s_w_res, s_w_res))

# Resize bounds of slice
    # resize height lower bounds
    r_h_lb = 640
    # resize height upper bounds
    r_h_ub = 640
    # resize height step
    r_h_res = 320
    # list of resize heights 
    r_h = list(range(r_h_lb, r_h_ub+r_h_res, r_h_res))

    # resize width lower bounds
    r_w_lb = 640
    # resize width upper bounds
    r_w_ub = 960
    # resize width step
    r_w_res = 320
    # list of resize widths
    r_w = list(range(r_w_lb, r_w_ub+r_w_res, r_w_res)) 

ss = SearchSpace()
print(ss.s_h)
print(ss.s_w)
print(ss.r_h)
print(ss.r_w)
