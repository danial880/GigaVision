def draw_rectangle(img, bbox,
                   bbox_color=(255, 255, 255),
                   thickness=3, alpha=0.2,
                   add_weight=False):
    output = img.copy()
    overlay = img.copy()
    if add_weight:
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, thickness)
    return overlay

def add_label(img, label, bbox,
              pose_label = "",
              text_color_cat=(0, 0, 0),
              text_color_pose=(128, 0, 128),
              show_pose=False,
              cat_label_thickness=2,
              pose_label_thickness=2,
              catlbl_font_size=2,
              poselbl_font_size=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_text_size = cv2.getTextSize(label, font, catlbl_font_size, cat_label_thickness)
    pose_text_size = cv2.getTextSize(pose_label, font, poselbl_font_size, pose_label_thickness)
    labels_total_width = cat_text_size[0][0] + pose_text_size[0][0]
    bbox_width = bbox[2] - bbox[0]
    label_distance = 25
    
    while(bbox_width< labels_total_width):
        label_distance = 10
        if pose_label_thickness>1:
            pose_label_thickness = 1
        if cat_label_thickness>1:
            cat_label_thickness = 1
        if catlbl_font_size > poselbl_font_size:
            catlbl_font_size = round(catlbl_font_size-0.1, 1)
        elif poselbl_font_size > catlbl_font_size:
            poselbl_font_size = round(poselbl_font_size-0.1, 1)
        else:
            catlbl_font_size = round(catlbl_font_size-0.1,1)
            poselbl_font_size = round(poselbl_font_size-0.1, 1)

        if catlbl_font_size==0.5 or  poselbl_font_size==0.5:           
            break
        cat_text_size = cv2.getTextSize(label, font, catlbl_font_size, cat_label_thickness)
        pose_text_size = cv2.getTextSize(label, font, poselbl_font_size, pose_label_thickness)
        labels_total_width = cat_text_size[0][0] + pose_text_size[0][0]
    if show_pose:
        if labels_total_width > (1.5*bbox_width):
            cv2.putText(img, pose_label, (bbox[0] + 5, bbox[1] - cat_text_size[0][1]-label_distance), font,
                poselbl_font_size, text_color_pose, pose_label_thickness)
            cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 10), font, catlbl_font_size, text_color_cat,
                cat_label_thickness)
        else:
            cv2.putText(img, pose_label, (bbox[0] + cat_text_size[0][0]+label_distance, bbox[1] - 10), font,
                poselbl_font_size, text_color_pose, pose_label_thickness)
            cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 10), font, catlbl_font_size, text_color_cat,
                cat_label_thickness)
    else:
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 10), font, catlbl_font_size, text_color_cat,
            cat_label_thickness)
    return img

def bbox_distribution(all_areas, label):
    #all_areas = list(set(all_areas))
    %matplotlib inline
    bins = [96*96, 256*256, 512*512, 1024*1024]
    a = {}
    a['small(96)'] = len([i for i in all_areas if i < bins[0]])
    a['96x96'] = len([i for i in all_areas if i > bins[0] and i < bins[1]])
    a['256x256'] = len([i for i in all_areas if i > bins[1] and i < bins[2]])
    a['512x512'] = len([i for i in all_areas if i > bins[2] and i < bins[3]])
    a['big(1024)'] = len([i for i in all_areas if i > bins[3]])
    plt.xlabel("BBox Size", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.bar(list(a.keys()), a.values(), color='g')
    plt.title(label=label,
          fontsize=24,
          color="green")
    plt.show()
