how to use:

python main.py --video_path = path/to/video

example:

python main.py --video_path=./pose_emotion/videos/test_video.mp4

illustration:

在main.py所在目录下运行main.py文件，参数为待处理的视频文件所在路径
上述命令中video_name应替换为在pose_emotion/videos文件夹中保存的视频文件
输入：一个5秒的拍摄人走路的视频文件，文件格式为.mp4，摄像头捕捉的视频文件应存放于pose_emotion/videos文件夹
输出：4个角度下的表情预测，即main.py中main()的返回值