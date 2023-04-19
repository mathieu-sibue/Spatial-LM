
from typing import Optional, Tuple, Union


def _rule_polar(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and direction from src to dst bounding boxes
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and direction
    """
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0 # left-top point
    bottom = (rect_src[3] - rect_dst[1]) <= 0   # 
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 

    if rect_intersect:
        return 0,0, 0
    elif top and left:
        a, b = abs(rect_dst[2] - rect_src[0]), abs(rect_dst[3] - rect_src[1])
        return int(a),int(b), 4
    elif left and bottom:
        a, b = abs(rect_dst[2] - rect_src[0]), abs(rect_dst[1] - rect_src[3])
        return int(a),int(b), 6
    elif bottom and right:
        a, b = abs(rect_dst[0] - rect_src[2]), abs(rect_dst[1] - rect_src[3])
        return int(a),int(b),8
    elif right and top:
        a, b = abs(rect_dst[0] - rect_src[2]), abs(rect_dst[3] - rect_src[1])
        return int(a),int(b), 2
    elif left:
        return int(abs(rect_src[0] - rect_dst[2])),0, 5
    elif right:
        return int(abs(rect_dst[0] - rect_src[2])),0, 1
    elif bottom:
        return 0, int(abs(rect_dst[1] - rect_src[3])), 7
    elif top:
        return 0, int(abs(rect_src[1] - rect_dst[3])), 3
    else:
        print('why the relative position is no where?')

# for each bboxs
def _fully_spatial_matrix(bboxs, word_ids):
    spatial_matrix = []
    for i in range(len(bboxs)):
        rows = []
        for j in range(len(bboxs)):
            if word_ids[i] is None or word_ids[j] is None:
                rows.append([0]*11)
            else:
                box1, box2 = bboxs[i],bboxs[j]
                d1,d2,direct = _rule_polar(box1,box2)
                # direct = angle //45 
                vect11 = [0]*9  # len = 9
                vect11[direct]=1    # len = 9
                vect11.append(1.0/(d1+1))   # len = 10
                vect11.append(1.0/(d2+1))   # len = 11
                rows.append(vect11)
        spatial_matrix.append(rows)
        # print(spatial_matrix)

    return spatial_matrix

if __name__ == '__main__':
    pass
