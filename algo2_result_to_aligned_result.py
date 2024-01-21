import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def algo2_result_to_aligned_result(
    result,
    r=0.1,
    dist_thresh=30,
    threshold=0.5,
):
    aligned_result = []

    for boxes in result:
        confident_boxes = boxes[boxes[:, 4] > threshold]

        refined_boxes = set()
        # confident_boxes = confident_boxes.copy()
        for i, cell in enumerate(confident_boxes):
            w_cell = cell[2] - cell[0]
            h_cell = cell[3] - cell[1]
            for v_i, (x_i, y_i) in enumerate(
                (
                    (cell[0], cell[1]),
                    (cell[2], cell[1]),
                    (cell[2], cell[3]),
                    (cell[0], cell[3]),
                )
            ):
                x_offset1 = max(w_cell * r, 4.0)
                y_offset1 = max(h_cell * r, 4.0)

                keep_x = []
                keep_y = []
                idx_i_j = []
                vertex_idx = []

                for j, another_cell in enumerate(confident_boxes):
                    if i != j:
                        x_offset2 = max((another_cell[2] - another_cell[0]) * r, 4.0)
                        y_offset2 = max((another_cell[3] - another_cell[1]) * r, 4.0)

                        for v_j, (x_j, y_j) in enumerate(
                            (
                                (another_cell[0], another_cell[1]),
                                (another_cell[2], another_cell[1]),
                                (another_cell[2], another_cell[3]),
                                (another_cell[0], another_cell[3]),
                            )
                        ):
                            xdist = abs(x_j - x_i)
                            ydist = abs(y_j - y_i)
                            dist = np.sqrt(xdist**2 + ydist**2)

                            if not (
                                xdist > x_offset1
                                or xdist > x_offset2
                                or ydist > y_offset1
                                or ydist > y_offset2
                                or dist > dist_thresh
                            ):
                                keep_x.append(x_j)
                                keep_y.append(y_j)
                                idx_i_j.append(j)
                                vertex_idx.append(v_j)

                            # if (
                            #     i == 62
                            #     and v_i == 0
                            #     and j == 60
                            #     and v_j == 1
                            #     or j == 62
                            #     and v_j == 0
                            #     and i == 60
                            #     and v_i == 1
                            # ):
                            #     print(
                            #         f"i = {i}, j = {j}, v_i = {v_i}, v_j = {v_j}"
                            #     )
                            # print("h_cell = ", h_cell)
                            # print("w_cell = ", w_cell)
                            # print("cell[2] - cell[0] ", cell[2] - cell[0])
                            # print("cell[3] - cell[1] ", cell[3] - cell[1])
                            # print("xdist < x_offset1   ", xdist, x_offset1)
                            # print("xdist < x_offset2   ", xdist, x_offset2)
                            # print("ydist < y_offset1   ", ydist, y_offset1)
                            # print("ydist < y_offset2   ", ydist, y_offset2)
                            # print("dist  < dist_thresh ", dist, dist_thresh)
                # if keep_x:
                keep_x.append(x_i)
                keep_y.append(y_i)
                idx_i_j.append(i)
                vertex_idx.append(v_i)

                mean_x = int(np.mean(keep_x))
                mean_y = int(np.mean(keep_y))

                # print(f"keep_x = {keep_x},  mean = {mean_x} ")
                # print(f"keep_y = {keep_y},  mean = {mean_y} ")
                # print("idx_i_j = ", idx_i_j)
                # print("vertex_idx = ", vertex_idx)
                # print("#" * 100)
                # print(confident_boxes.shape)

                # if 60 in idx_i_j or 62 in idx_i_j:
                #     print("keep_x = ", keep_x)
                #     print("keep_y = ", keep_y)
                #     print("idx_i_j = ", idx_i_j)
                #     print("vertex_idx = ", vertex_idx)
                #     print("#" * 100)

                for idx, v_idx in zip(idx_i_j, vertex_idx):
                    refined_boxes.add(int(str(v_idx) + str(idx)))
                    if v_idx == 0:
                        confident_boxes[idx, 0] = mean_x
                        confident_boxes[idx, 1] = mean_y
                    elif v_idx == 1:
                        confident_boxes[idx, 2] = mean_x
                        confident_boxes[idx, 1] = mean_y
                    elif v_idx == 2:
                        confident_boxes[idx, 2] = mean_x
                        confident_boxes[idx, 3] = mean_y
                    elif v_idx == 3:
                        confident_boxes[idx, 0] = mean_x
                        confident_boxes[idx, 3] = mean_y
                    else:
                        pass
                        # print(f"strange vertex idx{idx}")

        aligned_result.append(confident_boxes)

        # print("refined_boxes = ", refined_boxes)
        # print("len refined_boxes = ", len(refined_boxes))
    return aligned_result
