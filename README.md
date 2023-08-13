# Benchmarking of inference tools

Comparisson of the following inference tools:
 - Torch (a torch model without conversions)
 - Torch Script
 - Torch Trace
 - ONNX
 - OpenVino
 - TensorRT

## Results

The table shows average duration of 1 batch processing in seconds.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>batch_sz</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
      <th>8</th>
      <th>16</th>
    </tr>
    <tr>
      <th>platform</th>
      <th>tool</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">1GPU</th>
      <th>ONNX</th>
      <td>0.002859</td>
      <td>0.002777</td>
      <td>0.004151</td>
      <td>0.007069</td>
      <td>0.011151</td>
    </tr>
    <tr>
      <th>OpenVino</th>
      <td>0.052132</td>
      <td>0.089428</td>
      <td>0.178273</td>
      <td>0.351515</td>
      <td>0.699047</td>
    </tr>
    <tr>
      <th>TensorRT16</th>
      <td>0.002287</td>
      <td>0.013059</td>
      <td>0.013825</td>
      <td>0.014197</td>
      <td>0.014572</td>
    </tr>
    <tr>
      <th>TensorRT32</th>
      <td>0.004407</td>
      <td>0.006015</td>
      <td>0.009285</td>
      <td>0.014662</td>
      <td>0.020792</td>
    </tr>
    <tr>
      <th>Torch</th>
      <td>0.039872</td>
      <td>0.073134</td>
      <td>0.141640</td>
      <td>0.280348</td>
      <td>0.562738</td>
    </tr>
    <tr>
      <th>TorchScript</th>
      <td>0.004247</td>
      <td>0.005525</td>
      <td>0.007880</td>
      <td>0.010182</td>
      <td>0.015301</td>
    </tr>
    <tr>
      <th>TorchTrace</th>
      <td>0.004196</td>
      <td>0.005484</td>
      <td>0.008852</td>
      <td>0.010419</td>
      <td>0.015768</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">4cpu</th>
      <th>ONNX</th>
      <td>0.018597</td>
      <td>0.036123</td>
      <td>0.071911</td>
      <td>0.141841</td>
      <td>0.281526</td>
    </tr>
    <tr>
      <th>OpenVino</th>
      <td>0.054006</td>
      <td>0.092741</td>
      <td>0.183270</td>
      <td>0.365141</td>
      <td>0.729367</td>
    </tr>
    <tr>
      <th>Torch</th>
      <td>0.040505</td>
      <td>0.067429</td>
      <td>0.140837</td>
      <td>0.273973</td>
      <td>0.575115</td>
    </tr>
    <tr>
      <th>TorchScript</th>
      <td>0.039361</td>
      <td>0.064719</td>
      <td>0.139555</td>
      <td>0.267235</td>
      <td>0.549520</td>
    </tr>
    <tr>
      <th>TorchTrace</th>
      <td>0.039563</td>
      <td>0.066081</td>
      <td>0.136957</td>
      <td>0.280754</td>
      <td>0.551547</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">8cpu</th>
      <th>ONNX</th>
      <td>0.009528</td>
      <td>0.018458</td>
      <td>0.036521</td>
      <td>0.070518</td>
      <td>0.139202</td>
    </tr>
    <tr>
      <th>OpenVino</th>
      <td>0.027345</td>
      <td>0.046816</td>
      <td>0.091285</td>
      <td>0.179826</td>
      <td>0.360166</td>
    </tr>
    <tr>
      <th>Torch</th>
      <td>0.021856</td>
      <td>0.034159</td>
      <td>0.070351</td>
      <td>0.143785</td>
      <td>0.290883</td>
    </tr>
    <tr>
      <th>TorchScript</th>
      <td>0.020551</td>
      <td>0.033005</td>
      <td>0.070299</td>
      <td>0.139498</td>
      <td>0.279364</td>
    </tr>
    <tr>
      <th>TorchTrace</th>
      <td>0.020515</td>
      <td>0.032861</td>
      <td>0.069804</td>
      <td>0.134337</td>
      <td>0.281713</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">inteli5</th>
      <th>ONNX</th>
      <td>0.013898</td>
      <td>0.028126</td>
      <td>0.056938</td>
      <td>0.120684</td>
      <td>0.237966</td>
    </tr>
    <tr>
      <th>OpenVino</th>
      <td>0.017643</td>
      <td>0.034971</td>
      <td>0.065143</td>
      <td>0.126630</td>
      <td>0.305376</td>
    </tr>
    <tr>
      <th>Torch</th>
      <td>0.037988</td>
      <td>0.056516</td>
      <td>0.118128</td>
      <td>0.205616</td>
      <td>0.446461</td>
    </tr>
    <tr>
      <th>TorchScript</th>
      <td>0.030240</td>
      <td>0.055520</td>
      <td>0.106623</td>
      <td>0.246346</td>
      <td>0.460948</td>
    </tr>
    <tr>
      <th>TorchTrace</th>
      <td>0.039345</td>
      <td>0.057527</td>
      <td>0.115449</td>
      <td>0.272737</td>
      <td>0.462977</td>
    </tr>
  </tbody>
</table>

Сравнение метрик качества между моделями TensorRT с типами данных fp32 и fp16.
|model|AUROC|Accuracy|
|-----|-----|--------|
|fp32 |0.9417781|0.91496164|
|fp16|0.94175446|0.91496164|

## Install

### For CPU Experiments
Run command `make venv_cpu`

### For GPU Experiments
To prepare CUDA run command `make prep_cuda` and follow the instructions.
To prepare a python virtual environment run `venv_gpu`.

### For TensorRT experiments
It's quite difficult to install TensorRt, that's why the good way is to use Docker. So for tensorRT experiments run `make docker_lab`.
