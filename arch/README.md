# Architecture Diagrams

This folder contains detailed architecture diagrams for the LiDAR-Vision-VQA system.

## Planned Diagrams

The following diagrams will be added to illustrate the system architecture:

1. **Overall System Architecture**
   - End-to-end data flow from sensors to predictions
   - Component interactions and dependencies

2. **LiDAR Processing Pipeline**
   - Point cloud to BEV feature extraction (PCDet)
   - VATLiDAR transformer architecture
   - Geometric positional encodings and view embeddings

3. **Vision Processing Pipeline**
   - Multi-camera image encoding (CLIP + SAM)
   - VATVision transformer architecture
   - Spatial positional encodings and view embeddings

4. **Multimodal Fusion**
   - Sequence construction with delimiter tokens
   - LLM integration with LoRA adapters
   - Attention patterns across modalities

5. **Training & Inference Workflows**
   - Training loop with gradient accumulation
   - Checkpointing and resume logic
   - Inference pipeline with generation parameters

## Usage

Reference these diagrams in the main README with:

```markdown
![System Architecture](arch/system_architecture.png)
```

## Contributing

When adding new diagrams:
- Use consistent color schemes and styling
- Include labels for tensor dimensions
- Provide legends for component types
- Save in both PNG (for README) and source format (e.g., .drawio, .pptx)

### File Naming Conventions
- Use descriptive snake_case names: `system_architecture.png`, `vat_vision_detail.png`
- Include version numbers if iterating: `lidar_pipeline_v2.png`
- Keep source files with same base name: `system_architecture.drawio`

### Image Guidelines
- **Format**: PNG with transparent background where applicable
- **Dimensions**: Max 2000px width for detailed diagrams, 1200px for overview diagrams
- **Resolution**: 150 DPI minimum for clarity
- **File Size**: Optimize to keep under 500KB per image
