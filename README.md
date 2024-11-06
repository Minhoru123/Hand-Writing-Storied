# Handwriting OCR Analysis Project

This project implements an OCR (Optical Character Recognition) system for analyzing handwritten text snippets. The goal is to accurately detect and extract text from handwritten samples, providing confidence scores for each recognition.

### Key Features

* Dual OCR engine comparison (EasyOCR and Tesseract)
* Advanced image preprocessing pipeline
* Optimized parameter configurations
* Confidence score metrics
* CSV output format for results

## Technical Approach & Reasoning

### Why Two OCR Engines?

1. **EasyOCR**
   * Modern deep learning approach
   * Better suited for handwriting variation
   * More configurable parameters
   * Higher accuracy potential
   * Slower processing speed
2. **Tesseract**
   * Industry standard baseline
   * Faster processing
   * Good for comparison
   * Less accurate with handwriting

### Understanding Confidence Scores

* **Scale** : 0-1 (0% to 100% confidence)
* **Calculation Method** : Probability of correct text recognition
* **Interpretation** :

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs"></div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code><span><span>0.7-1.0: High confidence (likely correct)
  </span></span><span>0.4-0.7: Medium confidence (needs verification)
  </span><span><0.4: Low confidence (likely incorrect)</span></code></div></div></div></pre>

* **Project Results** :

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs"></div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code><span><span>Average Confidence: 0.101 (10.1%)
  </span></span><span>Best Cases: 0.963 (96.3%)</span></code></div></div></div></pre>

## Development Process

### Step 1: Initial Data Analysis

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">python</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-python"><span><span class="token"># Key findings from analysis</span><span>
</span></span><span><span>Image Characteristics</span><span class="token">:</span><span>
</span></span><span><span></span><span class="token">-</span><span> Dimensions</span><span class="token">:</span><span> 515x46 pixels </span><span class="token">(</span><span>average</span><span class="token">)</span><span>
</span></span><span><span></span><span class="token">-</span><span> Format</span><span class="token">:</span><span> RGB</span><span class="token">-</span><span>stored grayscale
</span></span><span><span></span><span class="token">-</span><span> Challenges</span><span class="token">:</span><span> Background artifacts</span><span class="token">,</span><span> variable text positioning</span></span></code></div></div></div></pre>

### Step 2: Preprocessing Pipeline Implementation

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">python</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-python"><span><span class="token">def</span><span></span><span class="token">preprocess_image</span><span class="token">(</span><span>self</span><span class="token">,</span><span> image</span><span class="token">)</span><span class="token">:</span><span>
</span></span><span><span></span><span class="token"># 1. Convert to grayscale</span><span>
</span></span><span><span>    gray </span><span class="token">=</span><span> cv2</span><span class="token">.</span><span>cvtColor</span><span class="token">(</span><span>image</span><span class="token">,</span><span> cv2</span><span class="token">.</span><span>COLOR_BGR2GRAY</span><span class="token">)</span><span>
</span></span><span>  
</span><span><span></span><span class="token"># 2. Enhance contrast (improve text visibility)</span><span>
</span></span><span><span>    clahe </span><span class="token">=</span><span> cv2</span><span class="token">.</span><span>createCLAHE</span><span class="token">(</span><span>clipLimit</span><span class="token">=</span><span class="token">2.0</span><span class="token">,</span><span> tileGridSize</span><span class="token">=</span><span class="token">(</span><span class="token">8</span><span class="token">,</span><span class="token">8</span><span class="token">)</span><span class="token">)</span><span>
</span></span><span><span>    enhanced </span><span class="token">=</span><span> clahe</span><span class="token">.</span><span class="token">apply</span><span class="token">(</span><span>gray</span><span class="token">)</span><span>
</span></span><span>  
</span><span><span></span><span class="token"># 3. Remove noise (clean up image)</span><span>
</span></span><span><span>    denoised </span><span class="token">=</span><span> cv2</span><span class="token">.</span><span>medianBlur</span><span class="token">(</span><span>enhanced</span><span class="token">,</span><span></span><span class="token">3</span><span class="token">)</span><span>
</span></span><span>  
</span><span><span></span><span class="token"># 4. Threshold (separate text from background)</span><span>
</span></span><span><span>    binary </span><span class="token">=</span><span> cv2</span><span class="token">.</span><span>adaptiveThreshold</span><span class="token">(</span><span>
</span></span><span><span>        denoised</span><span class="token">,</span><span></span><span class="token">255</span><span class="token">,</span><span>
</span></span><span><span>        cv2</span><span class="token">.</span><span>ADAPTIVE_THRESH_GAUSSIAN_C</span><span class="token">,</span><span>
</span></span><span><span>        cv2</span><span class="token">.</span><span>THRESH_BINARY</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">11</span><span class="token">,</span><span></span><span class="token">2</span><span>
</span></span><span><span></span><span class="token">)</span><span>
</span></span><span><span></span><span class="token">return</span><span> binary</span></span></code></div></div></div></pre>

### Step 3: OCR Engine Optimization

## EasyOCR Configuration Evolution

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">python</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-python"><span><span class="token"># Initial Parameters</span><span>
</span></span><span><span>default_params </span><span class="token">=</span><span></span><span class="token">{</span><span>
</span></span><span><span></span><span class="token">'paragraph'</span><span class="token">:</span><span></span><span class="token">False</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'min_size'</span><span class="token">:</span><span></span><span class="token">10</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'text_threshold'</span><span class="token">:</span><span></span><span class="token">0.7</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'link_threshold'</span><span class="token">:</span><span></span><span class="token">0.4</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'contrast_ths'</span><span class="token">:</span><span></span><span class="token">0.2</span><span>
</span></span><span><span></span><span class="token">}</span><span>
</span></span><span><span></span><span class="token"># Results: 28% success rate</span><span>
</span></span><span>
</span><span><span></span><span class="token"># Optimized Parameters</span><span>
</span></span><span><span>optimized_params </span><span class="token">=</span><span></span><span class="token">{</span><span>
</span></span><span><span></span><span class="token">'paragraph'</span><span class="token">:</span><span></span><span class="token">False</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'min_size'</span><span class="token">:</span><span></span><span class="token">5</span><span class="token">,</span><span></span><span class="token"># Smaller text detection</span><span>
</span></span><span><span></span><span class="token">'contrast_ths'</span><span class="token">:</span><span></span><span class="token">0.05</span><span class="token">,</span><span></span><span class="token"># More aggressive contrast</span><span>
</span></span><span><span></span><span class="token">'adjust_contrast'</span><span class="token">:</span><span></span><span class="token">1.5</span><span class="token">,</span><span></span><span class="token"># Enhanced contrast</span><span>
</span></span><span><span></span><span class="token">'text_threshold'</span><span class="token">:</span><span></span><span class="token">0.3</span><span class="token">,</span><span></span><span class="token"># Lower detection threshold</span><span>
</span></span><span><span></span><span class="token">'link_threshold'</span><span class="token">:</span><span></span><span class="token">0.2</span><span class="token">,</span><span></span><span class="token"># Lenient text linking</span><span>
</span></span><span><span></span><span class="token">'mag_ratio'</span><span class="token">:</span><span></span><span class="token">2</span><span></span><span class="token"># Increased image size</span><span>
</span></span><span><span></span><span class="token">}</span><span>
</span></span><span><span></span><span class="token"># Results: 30% success rate</span></span></code></div></div></div></pre>

## Tesseract Configuration Evolution

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">python</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-python"><span><span class="token"># Initial Config</span><span>
</span></span><span><span>default_config </span><span class="token">=</span><span></span><span class="token">{</span><span>
</span></span><span><span></span><span class="token">'lang'</span><span class="token">:</span><span></span><span class="token">'eng'</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'config'</span><span class="token">:</span><span></span><span class="token">'--psm 6'</span><span>
</span></span><span><span></span><span class="token">}</span><span>
</span></span><span><span></span><span class="token"># Results: 21% success rate</span><span>
</span></span><span>
</span><span><span></span><span class="token"># Optimized Config</span><span>
</span></span><span><span>optimized_config </span><span class="token">=</span><span></span><span class="token">{</span><span>
</span></span><span><span></span><span class="token">'lang'</span><span class="token">:</span><span></span><span class="token">'eng'</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'config'</span><span class="token">:</span><span></span><span class="token">'--psm 8 --oem 1'</span><span class="token">,</span><span>
</span></span><span><span></span><span class="token">'custom_oem_psm_config'</span><span class="token">:</span><span></span><span class="token">True</span><span>
</span></span><span><span></span><span class="token">}</span><span>
</span></span><span><span></span><span class="token"># Results: 22% success rate</span></span></code></div></div></div></pre>

### Step 4: Output Format Implementation

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">python</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-python"><span><span class="token"># CSV Structure</span><span>
</span></span><span><span>columns </span><span class="token">=</span><span></span><span class="token">[</span><span>
</span></span><span><span></span><span class="token">'snippet_name'</span><span class="token">,</span><span></span><span class="token"># Original filename</span><span>
</span></span><span><span></span><span class="token">'label'</span><span class="token">,</span><span></span><span class="token"># Recognized text</span><span>
</span></span><span><span></span><span class="token">'confidence_score'</span><span></span><span class="token"># 0-1 value</span><span>
</span></span><span><span></span><span class="token">]</span><span>
</span></span><span>
</span><span><span></span><span class="token"># Example Output</span><span>
</span></span><span><span>snippet_name</span><span class="token">,</span><span>label</span><span class="token">,</span><span>confidence_score
</span></span><span><span>image1</span><span class="token">.</span><span>png</span><span class="token">,</span><span>detected_text</span><span class="token">,</span><span class="token">0.78</span><span>
</span></span><span><span>image2</span><span class="token">.</span><span>png</span><span class="token">,</span><span class="token">,</span></span></code></div></div></div></pre>

## Implementation Details

### Project Structure

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs"></div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code><span><span>project_root/
</span></span><span>├── src/
</span><span>│   ├── enhanced_image_preprocessor.py
</span><span>│   ├── ocr_fine_tuner.py
</span><span>│   └── generate_csv.py
</span><span>├── data/
</span><span>│   └── snippets/
</span><span>└── output/
</span><span>    └── results.csv</span></code></div></div></div></pre>

### Key Components

1. **Image Preprocessor**
   * Handles image cleaning and enhancement
   * Implements complete preprocessing pipeline
   * Optimized for OCR input
2. **OCR Fine-tuner**
   * Manages OCR engine parameters
   * Implements testing configurations
   * Tracks performance metrics
3. **CSV Generator**
   * Creates standardized output format
   * Handles empty cases appropriately
   * Implements error handling

## Results & Analysis

### Performance Metrics

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs"></div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code><span><span>EasyOCR Performance:
</span></span><span>- Initial Success Rate: 28%
</span><span>- Optimized Success Rate: 30%
</span><span>- Average Confidence: 0.101
</span><span>- Best Case Confidence: 0.963
</span><span>- Processing Time: ~0.3s per image
</span><span>
</span><span>Tesseract Performance:
</span><span>- Initial Success Rate: 21%
</span><span>- Final Success Rate: 22%
</span><span>- Average Confidence: 0.009
</span><span>- Processing Time: ~0.2s per image</span></code></div></div></div></pre>

### Challenges & Solutions

1. **Low Confidence Scores**
   * Challenge: Poor initial recognition rates
   * Solution: Enhanced preprocessing and parameter tuning
   * Result: Improved high-confidence cases
2. **Processing Speed**
   * Challenge: Slow EasyOCR processing
   * Solution: Optimized parameters and preprocessing
   * Result: Reduced processing time

## Installation & Usage

### Dependencies

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">bash</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-bash"><span><span class="token"># Create virtual environment</span><span>
</span></span><span>python -m venv myenv
</span><span><span>myenv</span><span class="token">\</span><span>Scripts</span><span class="token">\</span><span>activate  </span><span class="token"># Windows</span><span>
</span></span><span><span></span><span class="token">source</span><span> myenv/bin/activate  </span><span class="token"># Linux/Mac</span><span>
</span></span><span>
</span><span><span></span><span class="token"># Install required packages</span><span>
</span></span><span><span>pip </span><span class="token">install</span><span> easyocr opencv-python numpy pandas</span></span></code></div></div></div></pre>

### Running the Project

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">bash</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 hover:bg-bg-200 opacity-60 hover:opacity-100"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-bash"><span><span class="token"># 1. Preprocess images</span><span>
</span></span><span>python src/enhanced_image_preprocessor.py
</span><span>
</span><span><span></span><span class="token"># 2. Generate final CSV</span><span>
</span></span><span>python src/generate_csv.py</span></code></div></div></div></pre>

## Future Improvements

1. **Accuracy Enhancement**
   * Custom model training
   * Advanced preprocessing techniques
   * Ensemble approach
2. **Performance Optimization**
   * GPU acceleration
   * Batch processing
   * Parallel execution
3. **Feature Additions**
   * Web interface
   * Real-time processing
   * Automated testing
