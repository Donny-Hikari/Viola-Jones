# 
# mergerect.py
#   Merge face rects produced by FaceDetector.
# 
# Author : Donny
# 

import queue

class Rect:
    def __init__(self, l, t, r, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b

    def width(self):
        return self.r - self.l

    def height(self):
        return self.b - self.t

    def area(self):
        return self.width() * self.height()

def genRectFromList(list_):
    """

    Parameters
    ----------
    list_ : list or tuple of shape [l, t, w, h]

    Returns
    -------
    A Rect type object of the corresponding rectangle represented in the given list.
    """
    return Rect(
        list_[0],
        list_[1],
        list_[0]+list_[2],
        list_[1]+list_[3]
    )

def getOverlapRect(rect1, rect2):
    return Rect(
        max(rect1.l, rect2.l), # left
        max(rect1.t, rect2.t), # top
        min(rect1.r, rect2.r), # right
        min(rect1.b, rect2.b)  # bottom
    )

def mergeRects(rects, overlap_rate=0.9, min_overlap_cnt=8):
    Q = queue.Queue()
    last_update = 0
    last_access = 0

    for x, y, w, h in rects:
        Q.put([Rect(x, y, x+w, y+h), 1])
        last_update += 1

    while last_access < last_update:
        r1, oc1 = Q.get(); last_access += 1
        updated = False

        last_access_2 = last_access
        while last_access_2 < last_update:
            r2, oc2 = Q.get(); last_access_2 += 1
            a1 = r1.area()
            a2 = r2.area()
            # get overlap rect
            ro = getOverlapRect(r1, r2)
            # get overlap area
            ao = ro.area()
            if ao >= min(a1, a2) * overlap_rate:
                # get merged rect
                # mr = Rect(
                #     min(r1.l, r2.l), # left
                #     min(r1.t, r2.t), # top
                #     max(r1.r, r2.r), # right
                #     max(r1.b, r2.b) # bottom
                # )
                mr = Rect(
                    (r1.l + r2.l) / 2, # left
                    (r1.t + r2.t) / 2, # top
                    (r1.r + r2.r) / 2, # right
                    (r1.b + r2.b) / 2 # bottom
                )

                last_access += 1 # r2 removed
                Q.put([mr, oc1+oc2]); last_update += 1
                updated = True
                break
            Q.put([r2, oc2])

        if not updated:
            if oc1 >= min_overlap_cnt:
                Q.put([r1, oc1])
    
    mergedRects = []
    while not Q.empty():
        rect, _ = Q.get()
        mergedRects.append([
            int(rect.l),
            int(rect.t),
            int(rect.width()),
            int(rect.height())
        ])
    return mergedRects