"""Generate synthetic RTI input; contains no real photographs or locations."""
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms

def generate(folder, count, width, height, seed):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx / max(width-1, 1) - 0.5) * 0.7
    ny = (yy / max(height-1, 1) - 0.5) * 0.7
    nz = np.sqrt(np.maximum(0, 1-nx*nx-ny*ny))
    texture = .32 + .10*np.sin(xx*.19)*np.cos(yy*.13) + .04*rng.random((height,width))
    albedo = np.stack([texture*1.1, texture*.9+.04, texture*.7+.07], axis=-1)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif()
    exif[271] = "Synthetic benchmark"
    exif[272] = "Generated data"
    exif[34665] = {40961:1,40962:width,40963:height,41989:50}
    lines = [str(count)]
    for i in range(count):
        z=.12+.84*(i+.5)/count
        a=i*math.pi*(3-math.sqrt(5))
        radius=math.sqrt(1-z*z)
        lx,ly=radius*math.cos(a),radius*math.sin(a)
        brightness=np.maximum(.03, nx*lx+ny*ly+nz*z)
        linear=np.clip(albedo*brightness[...,None]+.025,0,1)
        encoded=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)
        pixels=np.clip(np.rint(encoded*255),0,255).astype(np.uint8)
        name=f"synthetic_{i:05d}.jpg"
        Image.fromarray(pixels).save(folder/name,quality=98,subsampling=0,exif=exif,icc_profile=icc)
        lines.append(f"{name} {lx:.9f} {ly:.9f} {z:.9f}")
    (folder/"lights.lp").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (folder/"fixture.json").write_text(json.dumps({"synthetic":True,"images":count,"width":width,"height":height,"seed":seed,"quality":98,"icc":"sRGB"},indent=2))
    print(f"Generated {count} synthetic {width}x{height} images",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--images",type=int,default=512)
    parser.add_argument("--width",type=int,default=128)
    parser.add_argument("--height",type=int,default=96)
    parser.add_argument("--seed",type=int,default=777)
    args=parser.parse_args()
    if min(args.images,args.width,args.height)<1:parser.error("dimensions must be positive")
    generate(args.output,args.images,args.width,args.height,args.seed)
