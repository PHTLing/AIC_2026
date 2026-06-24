- Muốn chạy baseline cần clone TransNet về
```
git clone https://github.com/soCzech/TransNetV2.git
```

- Cấu trúc file metadata
```
{
    "video_id": "L28_V001",
    "type": "video",
    "video_path": "L28/L28_V001.mp4",
    "keyframes_folder_kpath": f"L28/L28_V001_keyframes", # Bổ sung dòng này
    "metadata_path": f"L28/L28_V001.json",
    "fps": 25.0,
    "segments": [
        {
            "segment_id": "seg_0001",
            "start_time": 0.0,
            "end_time": 3.92,
            "segment_caption": "...",
            "speech": [], # Speech của segment
            "keyframe": {
                "L28_V001_seg0001_00000.jpg": {
                    "object": [], # Object trong frame
                    "ocr": []     # Text trong frame
                },
                "....": {
                    "object": [],
                    "ocr": []
                }
            }
        },
        {
            "segment_id": "seg_0002",
            ...
        }
    ]
}
```