import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def dbscan_result_to_aligned_result(result, threshold=0.5):
    aligned_result = []

    for boxes in result:
        confident_boxes = boxes[boxes[:, 4] > threshold]
        if confident_boxes.shape[0] > 0:
            count_confident_boxes = confident_boxes.shape[0]

            updated_confident_boxes = pd.DataFrame(confident_boxes, columns=["x1", "y1", "x2", "y2", "confidance"])

            X = confident_boxes[:, [0, 2]].flatten(order="F")
            Y = confident_boxes[:, [1, 3]].flatten(order="F")

            X_scaled = MinMaxScaler().fit_transform(X.reshape((-1, 1)))
            Y_scaled = MinMaxScaler().fit_transform(Y.reshape((-1, 1)))

            min_samples = int(np.sqrt(count_confident_boxes) / 2)
            if min_samples < 2:
                min_samples = 2
            # print("min_samples = ", min_samples)
            clustering_X = DBSCAN(eps=0.01, min_samples=min_samples).fit_predict(X_scaled)
            clustering_Y = DBSCAN(eps=0.01, min_samples=min_samples).fit_predict(Y_scaled)

            labels_X = set(clustering_X)
            labels_Y = set(clustering_Y)

            updated_confident_boxes["label_x1"] = clustering_X[:count_confident_boxes]
            updated_confident_boxes["label_y1"] = clustering_Y[:count_confident_boxes]
            updated_confident_boxes["label_x2"] = clustering_X[count_confident_boxes:]
            updated_confident_boxes["label_y2"] = clustering_Y[count_confident_boxes:]

            updated_confident_boxes["x1_upd"] = 0
            updated_confident_boxes["y1_upd"] = 0
            updated_confident_boxes["x2_upd"] = 0
            updated_confident_boxes["y2_upd"] = 0

            # img = Image.open(img_path)
            # draw = ImageDraw.Draw(img)
            # colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255)]
            # width = 5

            x_correct = []
            y_correct = []
            for x in labels_X:
                if x != -1:
                    x_mean = int(np.mean(X[np.array(clustering_X) == x]))
                    x_correct.append(x_mean)

                    updated_confident_boxes.loc[updated_confident_boxes["label_x1"] == x, "x1_upd"] = x_mean
                    updated_confident_boxes.loc[updated_confident_boxes["label_x2"] == x, "x2_upd"] = x_mean
            for y in labels_Y:
                if y != -1:
                    y_mean = int(np.mean(Y[np.array(clustering_Y) == y]))
                    y_correct.append(y_mean)

                    updated_confident_boxes.loc[updated_confident_boxes["label_y1"] == y, "y1_upd"] = y_mean
                    updated_confident_boxes.loc[updated_confident_boxes["label_y2"] == y, "y2_upd"] = y_mean

            # for x in labels_X:
            #     if x != -1:
            #         x_mean = int(np.mean(X[np.array(clustering_X) == x]))
            #         endpoints = (x_mean, max(y_correct)), (x_mean, min(y_correct))
            #         draw.line(endpoints, fill=colors[0], width=width)
            # for y in labels_Y:
            #     if y != -1:
            #         y_mean = int(np.mean(Y[np.array(clustering_Y) == y]))
            #         endpoints = (max(x_correct), y_mean), (min(x_correct), y_mean)
            #         draw.line(endpoints, fill=colors[0], width=width)

            # print(f"sorted_confident_x = {sorted(x_correct)}")
            # print(f"sorted_confident_y = {sorted(y_correct)}")

            # img.save("/home/aiarhipov/centernet/imgs/output_lines.jpg")
            res = updated_confident_boxes[["x1_upd", "y1_upd", "x2_upd", "y2_upd", "confidance"]].to_numpy()
            res = res[(res[:, 0] < res[:, 2]) & (res[:, 1] < res[:, 3])]
            aligned_result.append(res)

    return aligned_result
