"""End-to-end CPU/GPU benchmark, with synthetic data and numerical comparison."""
import argparse, json, math, os, re, statistics, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image

def load_output(folder):
    info=json.loads((folder/"info.json").read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
    for key in ("width","height","nplanes","type","colorspace","materials"):
        if key not in info:raise ValueError(f"Missing {key}")
    material=info["materials"][0]
    arrays=[];profiles=[]
    for i in range((info["nplanes"]+2)//3):
        with Image.open(folder/f"plane_{i}.jpg") as image:
            image.load();assert image.size==(info["width"],info["height"])
            arrays.append(np.asarray(image.convert("RGB"),dtype=np.float64))
            profiles.append(image.info.get("icc_profile",b""))
    values=np.concatenate(arrays,axis=2)[:,:,:info["nplanes"]]
    scale=np.array(material["scale"],dtype=np.float64)
    bias=np.array(material["bias"],dtype=np.float64)
    if not np.isfinite(scale).all() or not np.isfinite(bias).all():raise ValueError("Non-finite material")
    coefficients=(values/255-bias)*scale*255
    return info,values,coefficients,profiles

def compare(cpu,gpu):
    a,ap,ac,ai=load_output(cpu);b,bp,bc,bi=load_output(gpu)
    for key in ("width","height","nplanes","type","colorspace"):
        assert a[key]==b[key],key
    def canonical(profile):
        # ICC date-time (24:36) and optional profile ID (84:100) are metadata,
        # not transforms. Every other header byte and all tags must match.
        data=bytearray(profile)
        if len(data)<128:raise ValueError("Invalid ICC header")
        data[24:36]=bytes(12);data[84:100]=bytes(16)
        return bytes(data)
    assert all(ai) and all(bi),"Missing ICC profile"
    assert [canonical(p) for p in ai]==[canonical(p) for p in bi],"ICC color transforms differ"
    difference=np.abs(ac-bc)
    quantized=np.abs(ap-bp)
    result={"coefficient_abs_max":float(difference.max()),"coefficient_abs_mean":float(difference.mean()),
            "encoded_jpg_abs_max":float(quantized.max()),"encoded_jpg_abs_mean":float(quantized.mean()),"icc_equal_semantics":True,"icc_equal_bytes":ai==bi}
    # Compare reconstructed PTM intensities, rather than only normalized coefficient JPEGs.
    if a["type"] in ("ptm","hsh"):
        errors=[]
        for angle in np.linspace(0,2*np.pi,12,endpoint=False):
            x=.7*np.cos(angle);y=.7*np.sin(angle)
            if a["type"]=="ptm":
                weights=np.array([1,x,y,x*x,x*y,y*y])[:a["nplanes"]//3]
            else:
                # HSH basis mirrors Rti::lightWeightsHsh in the upstream source.
                z=math.sqrt(1-x*x-y*y);phi=math.atan2(y,x)
                t=math.cos(min(math.acos(z),math.pi/2-.01));r=math.sqrt(t-t*t)
                weights=np.array([1/math.sqrt(2*math.pi),math.sqrt(6/math.pi)*math.cos(phi)*r,
                    math.sqrt(3/(2*math.pi))*(-1+2*t),math.sqrt(6/math.pi)*r*math.sin(phi),
                    math.sqrt(30/math.pi)*math.cos(2*phi)*(-t+t*t),math.sqrt(30/math.pi)*math.cos(phi)*(-1+2*t)*r,
                    math.sqrt(5/(2*math.pi))*(1-6*t+6*t*t),math.sqrt(30/math.pi)*(-1+2*t)*r*math.sin(phi),
                    math.sqrt(30/math.pi)*(-t+t*t)*math.sin(2*phi)])[:a["nplanes"]//3]
            ra=(ac.reshape(*ac.shape[:2],-1,3)*weights[:,None]).sum(axis=2)
            rb=(bc.reshape(*bc.shape[:2],-1,3)*weights[:,None]).sum(axis=2)
            errors.append(np.abs(ra-rb))
        result["reconstruction_abs_max_0_255"]=float(np.max(errors))
        result["reconstruction_abs_mean_0_255"]=float(np.mean(errors))
        assert result["reconstruction_abs_max_0_255"]<2.0,result
    return result

def benchmark(binary,fixture,output,basis,planes,repeats,workers):
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["RELIGHT_PROFILE"]="1"
    runs=[]
    for repetition in range(repeats):
        # Alternate order to reduce cache/order bias; report every wall-clock sample.
        for backend in (["cpu","cuda"] if repetition%2==0 else ["cuda","cpu"]):
            folder=output/f"{basis}_{backend}_{repetition}"
            if folder.exists():raise ValueError(f"Output already exists: {folder}")
            command=[str(binary),"-b",basis,"-p",str(planes),"-s","32","-w",str(workers),"-g",backend,str(fixture),str(folder)]
            start=time.perf_counter();process=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            wall=time.perf_counter()-start
            (output/f"{basis}_{backend}_{repetition}.log").write_text(process.stdout,encoding="utf-8")
            if process.returncode:raise RuntimeError(process.stdout[-4000:])
            profiles=re.findall(r"\[profile\] (\{[^\n]+\})",process.stdout)
            profile=json.loads(profiles[-1]) if profiles else {}
            if backend=="cuda" and profile.get("gpu_projection_calls",0)<1:raise ValueError("GPU benchmark used CPU fallback")
            runs.append({"backend":backend,"repetition":repetition,"wall_seconds":wall,"profile":profile})
            (output/"timings.json").write_text(json.dumps(runs,indent=2),encoding="utf-8")
            print(json.dumps(runs[-1]),flush=True)
    comparison=compare(output/f"{basis}_cpu_0",output/f"{basis}_cuda_0")
    fixture_info=json.loads((fixture/"fixture.json").read_text())
    cpu=statistics.median(r["wall_seconds"] for r in runs if r["backend"]=="cpu")
    gpu=statistics.median(r["wall_seconds"] for r in runs if r["backend"]=="cuda")
    return {"fixture":fixture_info,"basis":basis,"planes":planes,"workers":workers,"runs":runs,
            "cpu_median_seconds":cpu,"gpu_median_seconds":gpu,"end_to_end_speedup":cpu/gpu,
            "comparison":comparison,"cache_policy":"Alternating order; no OS cache purge", "includes_full_export":True}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("binary",type=Path);parser.add_argument("fixture",type=Path)
    parser.add_argument("output",type=Path);parser.add_argument("--basis",choices=["ptm","hsh"],default="ptm")
    parser.add_argument("--planes",type=int,default=18);parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--workers",type=int,default=2);parser.add_argument("--result",type=Path,required=True)
    args=parser.parse_args();result=benchmark(args.binary,args.fixture,args.output,args.basis,args.planes,args.repeats,args.workers)
    args.result.parent.mkdir(parents=True,exist_ok=True);args.result.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["comparison"],indent=2));print("End-to-end speedup:",result["end_to_end_speedup"])
