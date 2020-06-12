from deepsegment2 import DeepSegment
segmenter = DeepSegment(checkpoint_path='/Users/trinhgiang/Downloads/deepsegment3/checkpoint',params_path='/Users/trinhgiang/Downloads/deepsegment3/params', utils_path='/Users/trinhgiang/Downloads/deepsegment3/utils')
# print(segmenter.segment('Tư vấn cho khách hàng về chữ ký số và các phần mềm bảo hiểm xã hội điện tử, hóa đơn điện tử, chữ ký số ..vv.'))
sent = '- Nghiên cứu, chế tạo và phát triển thiết bị quan trắc khí thải tự động đã được ứng dụng vào một số dự án. Email: giangtt.bkhn@gmail.com'
print(len(segmenter.segment_long(sent)))