import cv2
from matplotlib import pyplot as plt

# 加载图片,以彩色模式读取图片
# cv2.IMREAD_COLOR：读取一副 彩色 图片，图片的透明度会被忽略，默认为该值，实际取值为1；
# cv2.IMREAD_GRAYSCALE;以 灰度 模式读取一张图片，实际取值为0
# cv2.IMREAD_UNCHANGED：加载一副彩色图像，透明度不会被忽略。

# img = cv2.imread("../image_data/timg (2).jpg",flags=cv2.IMREAD_COLOR)
# cv2.imshow('image', img)
# cv2.imwrite("pic1.jpg",img)
# # 等待键盘输入，在执行后面操作。如果没有这一步，图片会一闪而过
# # 函数是一个键盘绑定函数（相当于让程序在这里挂起暂停执行），他接受一个单位为毫秒的时间，它等待指定时间的键盘事件，在指定时间内发生了键盘事件，程序继续执行，否则必须等到时间结束才能继续执行，参数如果为0表示等待无限长的事件。
# cv2.waitKey(0) # 如果没有cv2.waitKey()函数，图像不会显示
# #删除建立的全部窗口
# cv2.destroyAllWindows()


# 创建一个名为Image的窗口
# cv2.namedWindow("Image",cv2.WINDOW_NORMAL) # 初始设定函数标签是cv2.WINDOW_AUTOSIZE。但是如果你把标签改成cv2.WINDOW_NORMAL，你就可以调整窗口大小了
# img = cv2.imread("pic1.jpg", 1)
# # 展示图片。两个窗口名相同
# # cv2.imshow("Image",img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# img = cv2.imread("pic1.jpg", 0)  # 加载图片
# cv2.imshow("image", img)  # 显示图片
# k = cv2.waitKey(0) & 0xff  # 获取键盘输入 # &0xff代表64位的键盘输入
# if k == 27:  # 如果键盘输入时是，Esc
#     cv2.destroyAllWindows()  # 关闭所有窗口
# elif k == ord('s'):  # 如果键盘输入时是，s
#     cv2.imwrite("pic2.jpg", img)  # 保存图片
#     cv2.destroyAllWindows()  # 关闭所有窗口


# img = cv2.imread("pic1.jpg",1)
# """
# 在pic.jpg上画一条坐标（50,50）开始，（100,100）结束，颜色为（255,0,0）
# 线宽为10，线的类型是cv2.LSD_REFINE_ADV的一条线
# """
# cv2.line(img,(200,250),(400,250),color=(255,105,180),thickness=5,lineType=cv2.LSD_REFINE_ADV)
# # 显示图片
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 参数img：绘制矩形的图像。
#
# 参数pt1：矩形左上角的坐标（元组）
#
# 参数pt2：矩形右下角的坐标
#
# 参数color：矩形的边框颜色。
#
# 参数thickness：矩形的边框的厚度，如果是正的。代表矩形的边框的厚度，负值，如-1填充，意味着要画一个填满的矩形。
#
# 参数lineType：矩形边框的类型。见α线型
#
# 参数shift（改变）：点坐标中的小数位数

# img = cv2.imread("pic1.jpg",1)
# """
# # 在图片上画矩形
# 在图片上，画一个左上角坐标（50,100）右下角坐标（150,150）的矩形，该矩形为100*50
# """
# cv2.rectangle(img,pt1=(250,80),pt2=(360,220),color=(255,105,180), thickness=3)
# # 显示图片
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = cv2.imread("pic1.jpg",1)
# """
# 在图片上画圆:
# 圆点在坐标（250,150），半径为10，颜色为（255，0，0）,圆的厚度-1 ，
# """
# cv2.circle(img,center=(250,150),radius=50,color=(255,0,0),thickness= 2)
# # 显示图片
# cv2.imshow('image2',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img = cv2.imread("pic1.jpg")
font = cv2.FONT_HERSHEY_SIMPLEX # 字体类型
# font = cv2.FONT_HERSHEY_DUPLEX
"""
将OpenCV放在图像img，坐标为（10,300），字体类型为font，大小为4，颜色为白色
"""
cv2.putText(img,"OpenCV",(300,300),font,1,(255,105,180),2)
cv2.imshow("fdsa",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 参数img：图像
#
# 参数text：要绘制的文本字符串。
#
# 参数org：org图片中文本字符串的左下角。
#
# 参数fontFace：字体类型，请参阅#HersheyFonts。
#
# 参数fontScale：字体比例因子，乘以特定字体的基础大小     字体大小
#
# 参数color：文本颜色
#
# 参数thickness：用于绘制文本的线条的厚度
#
# 参数lineType：线条类型。看到#线型
#
# 参数bottomLeftOrigin：如果为真，图像数据来源在左下角。否则，它在左上角
