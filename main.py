#!/usr/bin/env python
import sys

import cv2
import numpy as np

from params_manager import ParamsManager


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def combine_image(images_list: list):
    '''
    画像を結合する
    '''
    combined_img_h = None
    for images in images_list:
        # BGR(OpenCV画像) から RGB画像に変換
        rgb_image = cv2.cvtColor(images["bgr_image"], cv2.COLOR_BGR2RGB)
        im_v = vconcat_resize_min([rgb_image, images["hsv_image"]])
        if combined_img_h is None:
            combined_img_h = im_v
        else:
            combined_img_h = hconcat_resize_min([combined_img_h, im_v])
    return combined_img_h


# メイン関数
def main():

    # 画像を読み込み
    images_list = []
    for i in range(1, len(sys.argv)):
        image_path = sys.argv[i]
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)        # OpenCV用のカラー並びに変換する
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR画像 -> HSV画像
        images_list.append({
            "bgr_image": image,
            "hsv_image": hsv_image
        })

    # トラックバーのコールバック関数は何もしない空の関数
    def nothing(x):
        pass

    # パラメータをロードする
    pm = ParamsManager()
    params = pm.load()
    h_min = params["H_min"] if 'H_min' in params else 0
    h_max = params["H_max"] if 'H_max' in params else 128
    s_min = params["S_min"] if 'S_min' in params else 128
    s_max = params["S_max"] if 'S_max' in params else 255
    v_min = params["V_min"] if 'V_min' in params else 128
    v_max = params["V_max"] if 'V_max' in params else 255

    # トラックバーを作るため，まず最初にウィンドウを生成
    cv2.namedWindow("HSV Separate Tool")

    # トラックバーの生成
    cv2.createTrackbar("H_min", "HSV Separate Tool", h_min, 179, nothing)
    cv2.createTrackbar("H_max", "HSV Separate Tool", h_max, 179, nothing)
    cv2.createTrackbar("S_min", "HSV Separate Tool", s_min, 255, nothing)
    cv2.createTrackbar("S_max", "HSV Separate Tool", s_max, 255, nothing)
    cv2.createTrackbar("V_min", "HSV Separate Tool", v_min, 255, nothing)
    cv2.createTrackbar("V_max", "HSV Separate Tool", v_max, 255, nothing)
    switch_title = '0 : OFF\n 1 : ON'
    cv2.createTrackbar(switch_title, 'HSV Separate Tool', 1, 1, nothing)
    save_switch_title = 'Params\n 1 : Save'
    cv2.createTrackbar(save_switch_title, 'HSV Separate Tool', 0, 1, nothing)

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # トラックバーの値を取る
        h_min = cv2.getTrackbarPos("H_min", "HSV Separate Tool")
        h_max = cv2.getTrackbarPos("H_max", "HSV Separate Tool")
        s_min = cv2.getTrackbarPos("S_min", "HSV Separate Tool")
        s_max = cv2.getTrackbarPos("S_max", "HSV Separate Tool")
        v_min = cv2.getTrackbarPos("V_min", "HSV Separate Tool")
        v_max = cv2.getTrackbarPos("V_max", "HSV Separate Tool")
        switch = cv2.getTrackbarPos(switch_title, 'HSV Separate Tool')
        save_switch = cv2.getTrackbarPos(save_switch_title, 'HSV Separate Tool')

        # パラメータを保存する
        if save_switch == 1:
            cv2.waitKey(200)
            cv2.setTrackbarPos(save_switch_title, 'HSV Separate Tool', 0)
            pm.save({
                'H_min': h_min,
                'H_max': h_max,
                'S_min': s_min,
                'S_max': s_max,
                'V_min': v_min,
                'V_max': v_max,
            })

        if switch == 0:
            combined_image = combine_image(images_list)
            cv2.imshow('HSV Separate Tool', combined_image)
        else:
            result_images_list = []
            for images in images_list:
                image = images["bgr_image"]
                hsv_image = images["hsv_image"]
                # inRange関数で範囲指定２値化 -> マスク画像として使う
                mask_image = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_max, s_max, v_max)) # HSV画像なのでタプルもHSV並び
                # bitwise_andで元画像にマスクをかける -> マスクされた部分の色だけ残る
                result_hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_image)
                # HSV画像をBGR画像に変換
                result_image_bgr = cv2.cvtColor(result_hsv_image, cv2.COLOR_HSV2BGR)
                result_images_list.append({
                    "bgr_image": result_image_bgr,
                    "hsv_image": result_hsv_image
                })

            # (X)ウィンドウに表示
            combined_image = combine_image(result_images_list)
            cv2.imshow('HSV Separate Tool', combined_image)
    
    cv2.destroyAllWindows()


# "python main.py"として実行された時だけ動く様にするおまじない処理
if __name__ == "__main__":      # importされると"__main__"は入らないので，実行かimportかを判断できる．
    main()    # メイン関数を実行