def sliding_frame(img, frame=0.75, step=0.25):
    length = round(frame * img.shape[0])
    width = round(frame * img.shape[1])

    for i in range(0, img.shape[0] - length + 1, round(img.shape[0] * step)):
        for j in range(0, img.shape[1] - width + 1, round(img.shape[1] * step)):
            yield img[i:i + length, j:j + width]
