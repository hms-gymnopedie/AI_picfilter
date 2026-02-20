# AI_picfilter Frontend UI/UX Design Document

## 1. Overview

AI_picfilter is an AI-powered image style learning and filter generation system. Users upload reference images for the AI to learn visual characteristics (color, lighting, texture, atmosphere), then apply those learned styles to other images or export them as `.cube` LUT files.

This document defines the user interface design, interaction flows, technology stack, and component architecture.

---

## 2. Tech Stack & Rationale

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Framework | **Next.js 14 (App Router)** | SSR/SSG for fast initial load, API routes for BFF pattern, built-in image optimization |
| Language | **TypeScript** | Type safety for complex AI response shapes, better DX with autocomplete |
| Styling | **Tailwind CSS** | Utility-first for rapid prototyping, responsive design, dark mode support |
| State Management | **Zustand** | Lightweight, minimal boilerplate, good for managing filter parameters and UI state |
| Image Processing (Client) | **WebGL / Canvas API** | Real-time LUT preview at 4K without server roundtrips |
| File Handling | **react-dropzone** | Mature drag-and-drop file upload with validation |
| Data Visualization | **Recharts** | Histogram and color distribution charts for style analysis |
| Internationalization | **next-intl** | Korean/English bilingual support with Next.js App Router integration |
| HTTP Client | **ky** | Lightweight fetch wrapper with retry, timeout, streaming support |
| Real-time | **Native EventSource (SSE)** | Server-Sent Events for style learning progress; simpler than WebSocket for unidirectional updates |

### Why Next.js over Vite + React SPA?

- Image optimization via `next/image` reduces payload for reference/result image galleries
- API routes serve as a Backend-for-Frontend layer, handling auth and request shaping
- SSR ensures fast first contentful paint for the main dashboard

### Why WebGL for Preview?

- 3D LUT application is a per-pixel color mapping operation -- ideal for GPU shaders
- Enables real-time 4K preview (<16ms) without server roundtrips
- Fragment shader applies trilinear interpolation on the LUT cube, matching professional software behavior

---

## 3. Screen Inventory

### 3.1 Home / Dashboard

```
+------------------------------------------------------------------+
|  [Logo] AI_picfilter              [My Filters] [Settings] [Lang] |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  | + New Style      |  | Recent Filter 1  |  | Recent Filter 2  |  |
|  |                  |  | [thumbnail]      |  | [thumbnail]      |  |
|  |  Upload ref      |  | "Warm Sunset"    |  | "Cool Film"      |  |
|  |  images to       |  | 3 styles         |  | 1 style          |  |
|  |  get started     |  | [Apply] [Export] |  | [Apply] [Export] |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
|  +------------------+  +------------------+                        |
|  | Recent Filter 3  |  | Recent Filter 4  |                       |
|  | [thumbnail]      |  | [thumbnail]      |                       |
|  | "Moody B&W"      |  | "Pastel Dream"   |                       |
|  | 2 styles         |  | 1 style          |                       |
|  | [Apply] [Export] |  | [Apply] [Export] |                       |
|  +------------------+  +------------------+                        |
+------------------------------------------------------------------+
```

**Purpose**: Entry point showing recent/saved filters with quick actions.

### 3.2 Style Learning (Upload & Train)

```
+------------------------------------------------------------------+
|  [<- Back]  Style Learning                          [Step 1 of 3] |
+------------------------------------------------------------------+
|                                                                    |
|  Step 1: Upload Reference Images                                   |
|  +--------------------------------------------------------------+  |
|  |                                                              |  |
|  |     +-------+  +-------+  +-------+  +-------+              |  |
|  |     | img 1 |  | img 2 |  | img 3 |  |  + Add|              |  |
|  |     | [x]   |  | [x]   |  | [x]   |  |       |              |  |
|  |     +-------+  +-------+  +-------+  +-------+              |  |
|  |                                                              |  |
|  |  Drag & drop images here, or click to browse                 |  |
|  |  Supported: JPG, PNG, TIFF  |  Max: 20 images               |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
|  Style Name: [____________________________]                        |
|                                                                    |
|  Learning Method:                                                  |
|  ( ) Image-Adaptive 3D LUT  -- Fast, real-time capable             |
|  (*) NILUT                   -- Multi-style blending, compact      |
|  ( ) D-LUT (Diffusion)      -- Highest quality .cube output       |
|                                                                    |
|                                      [Cancel]  [Start Learning ->] |
+------------------------------------------------------------------+
```

### 3.3 Learning Progress

```
+------------------------------------------------------------------+
|  [<- Back]  Learning "Warm Sunset"                  [Step 2 of 3] |
+------------------------------------------------------------------+
|                                                                    |
|  Status: Analyzing color distributions...                          |
|  [========================================>-----------]  72%       |
|                                                                    |
|  Elapsed: 1m 23s  |  Estimated remaining: 32s                     |
|                                                                    |
|  +----------------------------+  +-----------------------------+   |
|  | Color Distribution         |  | Style Features Detected     |   |
|  | [histogram chart]          |  |                             |   |
|  |                            |  | Warm tones:       High      |   |
|  |                            |  | Contrast:         Medium    |   |
|  |                            |  | Saturation:       High      |   |
|  |                            |  | Shadow detail:    Low       |   |
|  +----------------------------+  +-----------------------------+   |
|                                                                    |
|  Log:                                                              |
|  > Preprocessing 5 reference images...                             |
|  > Extracting Gram matrices from VGG-19 layers...                  |
|  > Training NILUT model (epoch 142/200)...                         |
|  > Current deltaE: 0.83                                            |
+------------------------------------------------------------------+
```

### 3.4 Filter Application (Main Workspace)

```
+------------------------------------------------------------------+
|  [<- Back]  Filter Studio                    [Save] [Export .cube] |
+------------------------------------------------------------------+
|                                                                    |
|  +---------------------------+  +----------------------------+     |
|  |                           |  |                            |     |
|  |      Original Image       |  |      Filtered Image        |     |
|  |                           |  |                            |     |
|  |      [drag to upload      |  |      [live preview]        |     |
|  |       or select]          |  |                            |     |
|  |                           |  |                            |     |
|  +---------------------------+  +----------------------------+     |
|                                                                    |
|  View Mode: [Side by Side] [Slider Overlay] [Toggle A/B]          |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  | Active Filter: Warm Sunset                                   |  |
|  |                                                              |  |
|  | Intensity:  [==========|======]  70%                         |  |
|  |                                                              |  |
|  | Style Blend (NILUT multi-style):                             |  |
|  | Style A "Golden Hour":  [========|========]  55%             |  |
|  | Style B "Film Grain":   [====|============]  30%             |  |
|  | Style C "Warm Shadow":  [==|==============]  15%             |  |
|  |                                                              |  |
|  | [+ Add Style to Blend]                                       |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
|  Quick Filters: [Warm Sunset] [Cool Film] [Moody B&W] [+]        |
+------------------------------------------------------------------+
```

### 3.5 Export / Download

```
+------------------------------------------------------------------+
|  Export Filter                                                     |
+------------------------------------------------------------------+
|                                                                    |
|  Filter: Warm Sunset (NILUT, 3 blended styles)                    |
|                                                                    |
|  Export Format:                                                    |
|  [x] .cube (3D LUT) -- Compatible with Photoshop, Premiere, etc. |
|  [ ] .png (Hald image)                                            |
|  [ ] .json (NILUT model weights)                                  |
|                                                                    |
|  LUT Size: [33x33x33 v]                                           |
|                                                                    |
|  Current Blend Snapshot:                                           |
|  Style A: 55% | Style B: 30% | Style C: 15%                      |
|                                                                    |
|  [Cancel]                                   [Download]             |
+------------------------------------------------------------------+
```

### 3.6 My Filters (Library)

```
+------------------------------------------------------------------+
|  My Filters                           [Search___] [Sort: Recent v]|
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  | [before/after]   |  | [before/after]   |  | [before/after]   |  |
|  | Warm Sunset      |  | Cool Film        |  | Moody B&W        |  |
|  | NILUT | 3 styles |  | 3D LUT           |  | D-LUT            |  |
|  | 2024-02-15       |  | 2024-02-10       |  | 2024-02-08       |  |
|  | [Apply][Edit]    |  | [Apply][Edit]    |  | [Apply][Edit]    |  |
|  | [Export][Delete]  |  | [Export][Delete]  |  | [Export][Delete]  |  |
|  +------------------+  +------------------+  +------------------+  |
+------------------------------------------------------------------+
```

---

## 4. User Flows

### 4.1 Core Flow: Learn Style & Apply Filter

```
[Dashboard]
    |
    v
[Click "+ New Style"]
    |
    v
[Upload Reference Images] --> drag & drop / file picker
    |                          (validate format, size, count)
    v
[Configure Learning]
    |-- Select method (3D LUT / NILUT / D-LUT)
    |-- Name the style
    v
[Start Learning] --> POST /api/styles/learn
    |
    v
[Learning Progress Screen]
    |-- SSE stream: /api/styles/{id}/progress
    |-- Real-time updates: progress %, deltaE, epoch info
    |-- Color histogram visualization
    v
[Learning Complete]
    |
    v
[Filter Studio]
    |-- Upload target image
    |-- Real-time preview via WebGL (client-side LUT application)
    |-- Adjust intensity slider
    |-- Blend multiple styles (NILUT only)
    v
[Satisfied with result?]
    |
    +-- Yes --> [Export .cube] or [Save to Library]
    |
    +-- No  --> [Adjust sliders] / [Try different filter]
```

### 4.2 Quick Apply Flow

```
[Dashboard] --> [Click existing filter]
    |
    v
[Filter Studio] --> [Upload target image]
    |
    v
[Live Preview] --> [Adjust & Export]
```

### 4.3 Style Blending Flow (NILUT)

```
[Filter Studio]
    |
    v
[Select primary style from library]
    |
    v
[Click "+ Add Style to Blend"]
    |
    v
[Select additional styles (up to 5)]
    |
    v
[Adjust blend sliders] --> WebGL real-time preview
    |                       (weights sent as uniforms to shader)
    v
[Export blended result as new .cube or save as preset]
```

---

## 5. Component Hierarchy

```
App (Next.js App Router)
|
+-- layout.tsx
|   +-- Header
|   |   +-- Logo
|   |   +-- Navigation (My Filters, Settings)
|   |   +-- LanguageSwitcher
|   |   +-- ThemeToggle (light/dark)
|   +-- Footer
|
+-- page.tsx (Dashboard)
|   +-- FilterGrid
|   |   +-- NewStyleCard
|   |   +-- FilterCard (x N)
|   |       +-- FilterThumbnail
|   |       +-- FilterMeta (name, method, date)
|   |       +-- FilterActions (Apply, Export, Delete)
|
+-- /learn/page.tsx (Style Learning)
|   +-- StepIndicator
|   +-- ImageUploader
|   |   +-- DropZone
|   |   +-- ImagePreviewGrid
|   |   |   +-- ImagePreviewItem (x N)
|   |   +-- UploadProgress
|   +-- LearningConfig
|   |   +-- StyleNameInput
|   |   +-- MethodSelector (radio group)
|   |   +-- MethodDescription
|   +-- ActionBar (Cancel, Start Learning)
|
+-- /learn/[id]/progress/page.tsx (Learning Progress)
|   +-- ProgressBar
|   +-- StatusMessage
|   +-- TimeEstimate
|   +-- AnalyticsPanel
|   |   +-- ColorHistogram (Recharts)
|   |   +-- StyleFeaturesList
|   +-- TrainingLog
|
+-- /studio/page.tsx (Filter Application)
|   +-- ImageCompareView
|   |   +-- OriginalCanvas
|   |   +-- FilteredCanvas (WebGL)
|   |   +-- ComparisonSlider
|   |   +-- ViewModeToggle
|   +-- FilterControls
|   |   +-- IntensitySlider
|   |   +-- StyleBlendPanel
|   |   |   +-- BlendSlider (x N)
|   |   |   +-- AddStyleButton
|   |   +-- QuickFilterBar
|   |       +-- FilterChip (x N)
|   +-- StudioToolbar (Save, Export)
|
+-- /studio/export/page.tsx (Export Dialog)
|   +-- ExportFormatSelector
|   +-- LUTSizeSelector
|   +-- BlendSnapshot
|   +-- DownloadButton
|
+-- /filters/page.tsx (My Filters Library)
|   +-- SearchBar
|   +-- SortSelector
|   +-- FilterGrid
|       +-- FilterCard (x N)
|
+-- Shared / Common Components
    +-- ui/
    |   +-- Button, Input, Slider, RadioGroup, Select
    |   +-- Modal, Toast, Tooltip
    |   +-- Card, Badge, Spinner
    +-- ImageDropZone
    +-- ProgressBar
    +-- ErrorBoundary
    +-- EmptyState
    +-- LoadingState
```

---

## 6. Key UI/UX Decisions

### 6.1 Real-time Preview Strategy

The filter preview runs entirely on the client via WebGL:

1. Server returns LUT data (flattened 3D array or texture) after learning completes
2. Client loads LUT as a 3D texture in a WebGL fragment shader
3. Target image is rendered as a textured quad
4. Shader performs trilinear interpolation to map each pixel through the LUT
5. Intensity/blend sliders update shader uniforms -- no server calls needed

This achieves <16ms at 4K resolution, meeting the 60fps real-time target.

### 6.2 Comparison Modes

Three modes for before/after comparison:

- **Side by Side**: Two canvases, synchronized zoom/pan
- **Slider Overlay**: Single canvas with a draggable vertical divider
- **Toggle A/B**: Tap/click to switch between original and filtered

### 6.3 Responsive Design Strategy

| Breakpoint | Layout |
|-----------|--------|
| Desktop (>1024px) | Side-by-side image comparison, full control panel |
| Tablet (768-1024px) | Stacked images, collapsible control panel |
| Mobile (<768px) | Single image with toggle, bottom sheet controls |

### 6.4 Accessibility

- All sliders have ARIA labels with current percentage values
- Image comparison modes are keyboard navigable (arrow keys for slider)
- Color contrast meets WCAG 2.1 AA (4.5:1 minimum for text)
- Progress updates announced via `aria-live` regions
- Export format descriptions available as tooltips with keyboard focus

### 6.5 Error States

| Scenario | UI Response |
|---------|-------------|
| Upload fails | Toast with retry option, files remain in dropzone |
| Learning fails | Error banner with error details, option to retry with different settings |
| WebGL not supported | Fallback to server-side preview with polling |
| Network lost during learning | Auto-reconnect SSE, show "Reconnecting..." badge |
| Export fails | Modal with error detail and manual download link |

### 6.6 Dark Mode

Full dark mode support via Tailwind `dark:` variants. Default follows system preference, user can override via ThemeToggle in header.

---

## 7. Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| First Contentful Paint | < 1.2s | SSR with Next.js, optimized font loading |
| Largest Contentful Paint | < 2.5s | `next/image` with lazy loading, priority hints for above-fold |
| Filter Preview Latency | < 16ms | WebGL shader, client-side LUT application |
| Bundle Size (initial) | < 150KB gzipped | Code splitting per route, lazy load WebGL/Recharts |
| Image Upload Start | < 100ms | Immediate thumbnail preview via `URL.createObjectURL` |

---

## 8. Directory Structure (Proposed)

```
src/
  app/
    layout.tsx
    page.tsx                    # Dashboard
    learn/
      page.tsx                  # Upload & Configure
      [id]/
        progress/
          page.tsx              # Learning Progress
    studio/
      page.tsx                  # Filter Application
      export/
        page.tsx                # Export Dialog
    filters/
      page.tsx                  # My Filters Library
    api/                        # BFF API routes (proxy to backend)
  components/
    ui/                         # Base UI primitives
    image/                      # ImageDropZone, ImagePreview, etc.
    filter/                     # FilterCard, FilterControls, etc.
    studio/                     # WebGL canvas, comparison views
    charts/                     # ColorHistogram, StyleFeatures
  hooks/
    useWebGLPreview.ts          # WebGL LUT shader management
    useStyleLearning.ts         # SSE connection for learning progress
    useFilterLibrary.ts         # CRUD operations for saved filters
    useImageUpload.ts           # Upload with progress tracking
  lib/
    webgl/
      lut-shader.ts             # Fragment/vertex shaders for LUT
      lut-texture.ts            # 3D texture loading utilities
    api-client.ts               # ky-based API client
  stores/
    filter-store.ts             # Zustand store for active filter state
    studio-store.ts             # Zustand store for studio UI state
  types/
    filter.ts                   # Filter, Style, LUT type definitions
    api.ts                      # API request/response types
  i18n/
    ko.json
    en.json
```
