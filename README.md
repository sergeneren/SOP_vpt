# SOP_vpt
SOP_vpt is a Houdini SOP level path tracing exercise based on Physically Based Rendering (more info at [pbrt.org](pbrt.org)). It is purely for educational purposes and designed to implement modern rendering algorithms such as: Monte Carlo Integration, Multiple importance sampling etc. The real end goal is to be able to render volumetrics on a grid (hence vpt: volumetric path tracer) 

## Getting Started
Either download the source as a zip file or right click to a desired location and use "git clone https://github.com/sergeneren/SOP_vpt" command with git bash

### Installing

Open Houdini "Command Line Tools", cd to directory containing "SOP_vpt.cpp" and use command 
```
hcustom  SOP_vpt.cpp
```
 This will create a file called "SOP_vpt.dll" in "~documents/houdini<HOUDINI_VERSION>/dso/" folder. SOP_vpt will be automatically loaded at first start of houdini. If you cant see SOP_vpt at SOP level please check that your HOUDINI_DSO_PATH environment is not a custom one and the dll is placed under right folder.  
 
### Tests
 
Please see the examples folder for test.  

## Author

* **Sergen Eren** - *website* [Vimeo](Vimeo.com/sergeneren)

## License
This project is licensed under GNU General Public License v3.0

## Acknowledgments
* [PBRT](https://github.com/mmp/pbrt-v3/) - *Big thanks to Matt Pharr, Wenzel Jakob and Greg Humphreys*
* [SmallPaint](https://users.cg.tuwien.ac.at/zsolnai/gfx/smallpaint/) - pbrt implementation by Károly Zsolnai-Fehér
* [TUWIEN Rendering](https://youtu.be/pjc1QAI6zS0) - *Youtube playlist by Károly Zsolnai-Fehér*
* [XAPKOHHEH](https://vimeo.com/189423315) - Vex raytracer
