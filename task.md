# Конвертация в  TensorRT/OpenVino
Ускоряем инференс нашей обученной классификационной модели с помощью [TensorRT](https://developer.nvidia.com/tensorrt) и [OpenVino](https://docs.openvino.ai/latest/home.html).
Для конвертации можно использовать любой способ, показанный на лекции. Код обучения можно взять, например, в [занятии по обучению сеток](https://gitlab.com/deepschool_group/deepschool-cvr-mar23/-/tree/main/week-03-lightning). 

#### Задание: 
1) Нужно сконвертить модель с вашим любимым размером батча и разрядностями fp32 и fp16.
2) Замерить скорость инференса исходной модели (на торче) на GPU и CPU
3) И tensorrt моделей(в fp32 и fp16) на том же размере батча
3) И еще OpenVino 
3) Проверить, изменилась ли точность для tensorrt модели в fp16 по сравнению с торчовой моделью.

В результате должна получиться табличка вида:

| Модель                  | Latency | Metric |
|-------------------------| --- | --- |
| Исходная, GPU (PyTorch) | - | - |
| Исходная, CPU (PyTorch) | - | - |
| TensorRT (fp32)         | - | - |
| TensorRT (fp16)         | - | - |
| OpenVino          | - | - |

А так же не забудь указать версию железа (видеокарта, процессор) и софта (torch, tensorrt, openvino).

В ревьюеры к домашке нужно добавить @gavrins и @ar.kravchuk. И оставить ссылку на MR в [табличке](https://docs.google.com/spreadsheets/d/15mvpbSFR3gyPcvLq026MUThmET9cjavgyGfd72GqEOk/edit?usp=sharing)
на листе hw-trt-openvino.
