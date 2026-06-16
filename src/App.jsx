import React, { useState, useRef, useEffect } from 'react';
import { Upload, Settings, Image as ImageIcon, Download, CheckCircle, AlertCircle, Trash2, SlidersHorizontal, Palette, Printer, Sparkles, Type } from 'lucide-react';
import { loadModels, processImage } from './utils/imageProcessor';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

const SIZES = {
  '1inch': { width: 295, height: 413, label: '1寸 (295x413)' },
  '2inch': { width: 413, height: 579, label: '2寸 (413x579)' },
  'custom': { width: 300, height: 400, label: '自定义' }
};

const BG_COLORS = [
  { value: 'transparent', label: '保持原底色 (或透明)', color: 'transparent' },
  { value: '#ffffff', label: '白底', color: '#ffffff' },
  { value: '#438edb', label: '蓝底', color: '#438edb' },
  { value: '#ff0000', label: '红底', color: '#ff0000' }
];

function App() {
  const [images, setImages] = useState([]);
  const [config, setConfig] = useState({
    sizePreset: '1inch',
    width: SIZES['1inch'].width,
    height: SIZES['1inch'].height,
    customWidthCm: 2.5,
    customHeightCm: 3.5,
    dpi: 300,
    bgColor: 'transparent',
    beautyFilter: false,
    watermark: '',
    printLayout: false,
    format: 'jpg',
    quality: 90
  });
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadModels().then(success => {
      if (success) setModelsReady(true);
    });
  }, []);

  const handleConfigChange = (key, value) => {
    setConfig(prev => {
      const newConfig = { ...prev, [key]: value };
      
      if (key === 'sizePreset' && value !== 'custom') {
        newConfig.width = SIZES[value].width;
        newConfig.height = SIZES[value].height;
      }
      
      // Auto-calculate pixel dimensions if in custom mode
      if (newConfig.sizePreset === 'custom' && ['sizePreset', 'customWidthCm', 'customHeightCm', 'dpi'].includes(key)) {
        newConfig.width = Math.round((newConfig.customWidthCm / 2.54) * newConfig.dpi) || 300;
        newConfig.height = Math.round((newConfig.customHeightCm / 2.54) * newConfig.dpi) || 400;
      }
      
      return newConfig;
    });
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    
    const newImages = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      originalUrl: URL.createObjectURL(file),
      status: 'idle',
      result: null,
      errorMsg: ''
    }));
    
    setImages(prev => [...prev, ...newImages]);
    
    // Clear input so same file can be uploaded again if needed
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeImage = (id) => {
    setImages(prev => prev.filter(img => img.id !== id));
  };

  const processAll = async () => {
    if (!modelsReady) {
      alert("AI模型正在加载，请稍候...");
      return;
    }
    
    setIsProcessing(true);
    
    const toProcess = images.filter(img => img.status === 'idle' || img.status === 'error');
    
    for (let i = 0; i < toProcess.length; i++) {
      const img = toProcess[i];
      
      setImages(prev => prev.map(item => 
        item.id === img.id ? { ...item, status: 'processing' } : item
      ));
      
      try {
        const result = await processImage(img.file, config);
        setImages(prev => prev.map(item => 
          item.id === img.id ? { ...item, status: 'success', result } : item
        ));
      } catch (error) {
        setImages(prev => prev.map(item => 
          item.id === img.id ? { ...item, status: 'error', errorMsg: error.message } : item
        ));
      }
    }
    
    setIsProcessing(false);
  };

  const downloadAll = async () => {
    const successfulImages = images.filter(img => img.status === 'success' && img.result);
    if (successfulImages.length === 0) return;

    if (successfulImages.length === 1) {
      saveAs(successfulImages[0].result.blob, successfulImages[0].result.fileName);
      return;
    }

    const zip = new JSZip();
    successfulImages.forEach(img => {
      zip.file(img.result.fileName, img.result.blob);
    });
    
    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, 'id_photos.zip');
  };

  return (
    <div className="app-container">
      <header>
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
          Smart ID Photo Pro
        </h1>
        <p className="text-slate-400 mt-2">基于纯前端 AI 的终极证件照处理工具 | 自动抠图 · 智能裁剪 · 一键排版</p>
      </header>

      <div className="main-grid">
        <div className="left-panel">
          <div className="glass-panel">
            <div 
              className="upload-area" 
              onClick={() => fileInputRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('drag-active'); }}
              onDragLeave={(e) => { e.currentTarget.classList.remove('drag-active'); }}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove('drag-active');
                const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
                if (files.length) {
                  const dataTransfer = new DataTransfer();
                  files.forEach(f => dataTransfer.items.add(f));
                  fileInputRef.current.files = dataTransfer.files;
                  const event = new Event('change', { bubbles: true });
                  fileInputRef.current.dispatchEvent(event);
                }
              }}
            >
              <Upload className="upload-icon" />
              <h3 className="text-xl font-semibold">点击或拖拽上传图片</h3>
              <p className="text-slate-400 mt-2 text-sm">支持 JPG, PNG 格式。支持批量上传几十张照片同时处理。</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                multiple 
                accept="image/*" 
                style={{ display: 'none' }} 
              />
            </div>

            {images.length > 0 && (
              <div className="batch-list mt-8">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="flex items-center gap-2 text-lg font-semibold">
                    <ImageIcon size={20} className="text-blue-400" /> 图片列表 ({images.length})
                  </h3>
                  <div className="flex gap-4">
                    <button className="btn btn-secondary text-sm px-4 py-2" onClick={() => setImages([])}>清空</button>
                    <button 
                      className="btn btn-primary text-sm px-6 py-2 shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50" 
                      onClick={processAll}
                      disabled={isProcessing || !modelsReady}
                    >
                      {isProcessing ? '处理中...' : '一键批量处理'}
                    </button>
                  </div>
                </div>
                
                <div className="flex flex-col gap-3 max-h-[500px] overflow-y-auto pr-2">
                  {images.map(img => (
                    <div key={img.id} className="batch-item bg-slate-800/50 p-3 rounded-lg border border-slate-700/50 flex items-center gap-4 hover:border-slate-600 transition-colors">
                      <img src={img.originalUrl} alt="preview" className="w-16 h-16 rounded object-cover bg-black/50 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate text-sm mb-1">{img.file.name}</div>
                        <div className="text-xs flex items-center gap-1.5">
                          {img.status === 'idle' && <span className="text-slate-400">等待处理</span>}
                          {img.status === 'processing' && <><Settings className="w-3.5 h-3.5 text-blue-400 animate-spin" /> <span className="text-blue-400">处理中 (可能涉及AI抠图)...</span></>}
                          {img.status === 'success' && <><CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> <span className="text-emerald-400">处理成功</span></>}
                          {img.status === 'error' && <><AlertCircle className="w-3.5 h-3.5 text-red-400" /> <span className="text-red-400 truncate" title={img.errorMsg}>{img.errorMsg}</span></>}
                        </div>
                      </div>
                      
                      {img.status === 'success' && (
                        <div className="shrink-0 relative group">
                          <img src={img.result.url} alt="result" className="w-16 h-16 rounded object-cover border-2 border-emerald-500 shadow-lg shadow-emerald-500/20" />
                          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded flex items-center justify-center cursor-pointer"
                               onClick={() => saveAs(img.result.blob, img.result.fileName)}>
                            <Download size={16} className="text-white" />
                          </div>
                        </div>
                      )}
                      
                      <button 
                        className="text-slate-500 hover:text-red-400 transition-colors shrink-0 p-2"
                        onClick={() => removeImage(img.id)}
                      >
                        <Trash2 size={18} />
                      </button>
                    </div>
                  ))}
                </div>
                
                {images.some(img => img.status === 'success') && (
                  <button className="btn btn-primary w-full mt-6 py-3 flex justify-center items-center gap-2 font-bold text-lg" onClick={downloadAll}>
                    <Download size={24} />
                    下载全部成功图片 (ZIP)
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="right-panel flex flex-col gap-4">
          <div className="glass-panel">
            <h3 className="flex items-center gap-2 text-lg font-semibold mb-6 border-b border-slate-700 pb-3">
              <SlidersHorizontal size={20} className="text-blue-400" /> 尺寸规格
            </h3>
            
            <div className="settings-group mb-5">
              <label className="settings-label text-sm text-slate-300 font-medium mb-2 block">规格预设</label>
              <select 
                className="select-input w-full bg-slate-900/50 border border-slate-700 rounded-lg p-3 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                value={config.sizePreset}
                onChange={(e) => handleConfigChange('sizePreset', e.target.value)}
              >
                {Object.entries(SIZES).map(([key, value]) => (
                  <option key={key} value={key} className="bg-slate-800">{value.label}</option>
                ))}
              </select>
            </div>

            {config.sizePreset === 'custom' && (
              <div className="settings-group custom-size-inputs mb-5">
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex-1">
                    <label className="settings-label text-xs text-slate-400 block mb-1">宽度 (cm)</label>
                    <input 
                      type="number" 
                      step="0.1"
                      className="text-input w-full bg-slate-900/50 border border-slate-700 rounded-lg p-2.5" 
                      value={config.customWidthCm}
                      onChange={(e) => handleConfigChange('customWidthCm', parseFloat(e.target.value))}
                    />
                  </div>
                  <span className="text-slate-500 mt-5">x</span>
                  <div className="flex-1">
                    <label className="settings-label text-xs text-slate-400 block mb-1">高度 (cm)</label>
                    <input 
                      type="number" 
                      step="0.1"
                      className="text-input w-full bg-slate-900/50 border border-slate-700 rounded-lg p-2.5" 
                      value={config.customHeightCm}
                      onChange={(e) => handleConfigChange('customHeightCm', parseFloat(e.target.value))}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <label className="settings-label text-xs text-slate-400 block mb-1">打印分辨率 (DPI)</label>
                    <input 
                      type="number" 
                      className="text-input w-full bg-slate-900/50 border border-slate-700 rounded-lg p-2.5" 
                      value={config.dpi}
                      onChange={(e) => handleConfigChange('dpi', parseInt(e.target.value))}
                    />
                  </div>
                  <div className="flex-1 mt-5 text-xs text-slate-500 text-right">
                    转换像素: <span className="text-emerald-400 font-mono">{config.width} x {config.height} px</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="glass-panel">
            <h3 className="flex items-center gap-2 text-lg font-semibold mb-6 border-b border-slate-700 pb-3">
              <Sparkles size={20} className="text-purple-400" /> 画面处理
            </h3>

            <div className="settings-group mb-5">
              <label className="settings-label text-sm text-slate-300 font-medium mb-2 flex items-center gap-2">
                <Palette size={16} /> 背景替换
              </label>
              <p className="text-xs text-slate-400 mb-3">使用 AI 自动抠图并换底色（纯前端抠图，保护隐私）</p>
              <div className="grid grid-cols-2 gap-2">
                {BG_COLORS.map(bg => (
                  <button
                    key={bg.value}
                    onClick={() => handleConfigChange('bgColor', bg.value)}
                    className={`flex items-center gap-2 p-2 rounded border transition-all ${config.bgColor === bg.value ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-slate-500 bg-slate-800/30'}`}
                  >
                    <div className="w-5 h-5 rounded-full border border-slate-600 shadow-inner" style={{ background: bg.value === 'transparent' ? 'repeating-conic-gradient(#333 0% 25%, transparent 0% 50%) 50% / 10px 10px' : bg.color }}></div>
                    <span className="text-sm">{bg.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="settings-group mb-5 flex items-center justify-between p-3 bg-slate-800/30 rounded-lg border border-slate-700/50">
              <div>
                <label className="settings-label text-sm text-slate-300 font-medium flex items-center gap-2 mb-1">
                  <Sparkles size={16} className="text-pink-400" /> 智能美颜
                </label>
                <p className="text-xs text-slate-400">轻微提亮肤色和柔化边缘</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" checked={config.beautyFilter} onChange={(e) => handleConfigChange('beautyFilter', e.target.checked)} />
                <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-pink-500"></div>
              </label>
            </div>

            <div className="settings-group mb-2">
              <label className="settings-label text-sm text-slate-300 font-medium mb-2 flex items-center gap-2">
                <Type size={16} /> 防盗水印
              </label>
              <input 
                type="text" 
                placeholder="例如: 仅供办理业务使用"
                className="text-input w-full bg-slate-900/50 border border-slate-700 rounded-lg p-2.5 text-sm" 
                value={config.watermark}
                onChange={(e) => handleConfigChange('watermark', e.target.value)}
              />
            </div>
          </div>

            <div className="glass-panel mt-4">
              <h3 className="flex items-center gap-2 text-lg font-semibold mb-6 border-b border-slate-700 pb-3">
                <Settings size={20} className="text-amber-400" /> 高级裁剪微调
              </h3>
              
              <div className="settings-group mb-5">
                <label className="settings-label text-sm flex justify-between mb-2">
                  <span>脸部占比 (控制裁剪拉近/拉远)</span>
                  <span className="text-blue-400 font-mono">{config.faceRatio || 45}%</span>
                </label>
                <input 
                  type="range" 
                  min="25" 
                  max="70" 
                  value={config.faceRatio || 45} 
                  className="w-full accent-blue-500"
                  onChange={(e) => handleConfigChange('faceRatio', parseInt(e.target.value))}
                />
                <p className="text-xs text-slate-500 mt-1">值越小，画面越广（露出更多肩膀）。标准证件照建议 40-50%。</p>
              </div>

              <div className="settings-group mb-5">
                <label className="settings-label text-sm flex justify-between mb-2">
                  <span>头顶留白 (控制上下偏移)</span>
                  <span className="text-blue-400 font-mono">{config.headroom || 60}%</span>
                </label>
                <input 
                  type="range" 
                  min="20" 
                  max="120" 
                  value={config.headroom || 60} 
                  className="w-full accent-blue-500"
                  onChange={(e) => handleConfigChange('headroom', parseInt(e.target.value))}
                />
                <p className="text-xs text-slate-500 mt-1">值越大，头顶上方的空间越多（适合头发蓬松的人）。</p>
              </div>
            </div>

            <div className="glass-panel mt-4">
              <h3 className="flex items-center gap-2 text-lg font-semibold mb-6 border-b border-slate-700 pb-3">
                <Printer size={20} className="text-emerald-400" /> 导出与打印
              </h3>

            <div className="settings-group mb-5 flex items-center justify-between p-3 bg-slate-800/30 rounded-lg border border-slate-700/50">
              <div>
                <label className="settings-label text-sm text-slate-300 font-medium flex items-center gap-2 mb-1">
                  <Printer size={16} /> 生成 4x6 打印排版
                </label>
                <p className="text-xs text-slate-400">将多张证件照铺满标准 6寸相纸</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" checked={config.printLayout} onChange={(e) => handleConfigChange('printLayout', e.target.checked)} />
                <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
              </label>
            </div>

            <div className="settings-group mb-5">
              <label className="settings-label text-sm text-slate-300 font-medium mb-2 block">图片格式</label>
              <select 
                className="select-input w-full bg-slate-900/50 border border-slate-700 rounded-lg p-2.5 text-sm"
                value={config.format}
                onChange={(e) => handleConfigChange('format', e.target.value)}
              >
                <option value="jpg" className="bg-slate-800">JPG (文件小，可调画质)</option>
                <option value="png" className="bg-slate-800">PNG (无损画质，适合打印)</option>
              </select>
            </div>

            {config.format === 'jpg' && (
              <div className="settings-group">
                <label className="settings-label text-sm flex justify-between mb-2">
                  <span>导出画质</span>
                  <span className="text-blue-400 font-mono">{config.quality}%</span>
                </label>
                <input 
                  type="range" 
                  min="10" 
                  max="100" 
                  step="1"
                  value={config.quality} 
                  className="w-full accent-blue-500"
                  onChange={(e) => handleConfigChange('quality', parseInt(e.target.value))}
                />
              </div>
            )}
            
            <div className="mt-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-2 h-2 rounded-full ${modelsReady ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></div>
                <h4 className="text-sm font-semibold text-slate-300">系统状态</h4>
              </div>
              <p className="text-xs text-slate-400 ml-4 leading-relaxed">
                {modelsReady ? 'AI 人脸识别就绪。抠图模型会在首次使用时自动下载 (约30MB)，此后将缓存于浏览器极速运行。' : '正在初始化本地 AI 引擎...'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
