"""
PPE input is the exact {track_id: worker} mapping returned by update_workers().
"""
import copy
import json
import math
from pathlib import Path
from zone_core import load_config, scaled_zones
from zone_localisation import ZoneLocaliser


class ZoneModule:
    """One instance per camera/video; reset by creating a new instance.

    Zone PPE status uses recently detected evidence, not proof of correct wear.
    Missing evidence -> REVIEW, not an automatically confirmed violation.
    """
    def __init__(self, config, camera_id, retention_seconds=.5,
                 boundary_margin_px=6, track_timeout_seconds=3.0):
        if not math.isfinite(retention_seconds) or retention_seconds <= 0:
            raise ValueError('retention_seconds must be finite and greater than zero.')
        if not math.isfinite(track_timeout_seconds) or track_timeout_seconds <= 0:
            raise ValueError('track_timeout_seconds must be finite and greater than zero.')
        self.config=copy.deepcopy(config)
        self.camera_id=camera_id
        self.retention_seconds=float(retention_seconds)
        self.boundary_margin_px=boundary_margin_px
        self.track_timeout_seconds=track_timeout_seconds
        self.localiser=None
        self.track_context={}
        self.last_timestamp=None

    @classmethod
    def from_json(cls, config_path, camera_id, **kwargs):
        return cls(load_config(config_path),camera_id,**kwargs)

    def process(self,workers,frame_size,video_time,ppe_last_seen=None):
        """Return copied workers enriched with zone_* results.

        workers: output of your friend's update_workers(), with bbox and ppe.
        frame_size: (width,height) of the clean original frame.
        video_time: frame_index/fps, or elapsed monotonic time for a live stream.
        ppe_last_seen: optional {track_id:{ppe_name:last_positive_timestamp}}.

        If histories are omitted, only the current worker['ppe'] is checked.
        Do not feed an entire frame's PPE list to an individual worker.
        """
        now=float(video_time)
        if not math.isfinite(now) or now<0:
            raise ValueError('video_time must be finite and non-negative.')
        if self.last_timestamp is not None and now<=self.last_timestamp:
            raise ValueError('Timestamps must increase; create a new ZoneModule for a new video.')
        if not isinstance(workers,dict):
            raise ValueError('workers must be the dictionary returned by update_workers().')
        frame_size=tuple(frame_size)
        if self.localiser is None:
            self.localiser=ZoneLocaliser(self.config,self.camera_id,frame_size,self.boundary_margin_px)
        elif frame_size!=self.localiser.frame_size:
            raise ValueError('Frame dimensions changed. Create a new ZoneModule.')
        # Validate geometry before changing temporal context.
        locations={tid:self.localiser.locate(w['bbox'],w.get('feet_visible')) for tid,w in workers.items()}
        self.last_timestamp=now
        for tid in list(self.track_context):
            if now-self.track_context[tid]['last_seen']>=self.track_timeout_seconds:
                del self.track_context[tid]
        output=copy.deepcopy(workers)
        for tid,worker in output.items():
            location=locations[tid]
            worker.update(location)
            worker['zone_camera_id']=self.camera_id
            worker['zone_timestamp_s']=now
            key=None if location['zone_ids'] is None else tuple(location['zone_ids'])
            context=self.track_context.get(tid)
            if context is None or context['zones']!=key or key is None:
                context={'zones':key,'entered':now,'last_seen':now}
                self.track_context[tid]=context
            context['last_seen']=now
            worker['zone_retained_ppe']=None
            worker['zone_not_detected']=None
            # Keep the friend's original status intact. Use this new status for
            # zone-specific display and decisions, instead of a fixed global zone.
            if key is None:
                status='UNCERTAIN'
            elif not key:
                status='UNMONITORED'
            elif location['access_prohibited']:
                status='UNSAFE'
            elif not location['required_ppe']:
                status='NO_PPE_REQUIRED'
                worker['zone_retained_ppe']=[]
                worker['zone_not_detected']=[]
            elif tid is None:
                status='UNCERTAIN'
            else:
                if ppe_last_seen is not None:
                    history=ppe_last_seen.get(tid)
                    evidence=None if history is None else {
                        item for item,seen in history.items()
                        if math.isfinite(float(seen)) and context['entered']<=float(seen)<=now
                        and now-float(seen)<self.retention_seconds
                    }
                else:
                    current=worker.get('ppe')
                    evidence=None if current is None else set(current)
                if evidence is None:
                    status='UNCERTAIN'
                else:
                    required=set(location['required_ppe'])
                    # Explicit negative classes and ambiguous ownership, if a
                    # detector exposes them, must not support a positive label.
                    conflicts=set(worker.get('violations',[])) | set(worker.get('ambiguous_ppe',[]))
                    evidence-=conflicts
                    missing=sorted(required-evidence)
                    worker['zone_retained_ppe']=sorted(evidence)
                    worker['zone_not_detected']=missing
                    status='REVIEW' if missing else 'REQUIRED_PPE_DETECTED'
                    if required & set(worker.get('ambiguous_ppe',[])):
                        status='UNCERTAIN'
            worker['zone_compliance_status']=status
        return output

    def annotate(self,frame,workers):
        """Return an overlay copy. Original frame remains clean for inference."""
        import cv2
        if self.localiser is None:
            raise ValueError('Call process() before annotate().')
        output=self.localiser.draw(frame)
        height,width=frame.shape[:2]
        for tid,w in workers.items():
            x1,y1,x2,y2=map(int,w['bbox'])
            colour=(0,190,0) if w['zone_compliance_status'] in ('REQUIRED_PPE_DETECTED','NO_PPE_REQUIRED') else (0,160,255)
            cv2.rectangle(output,(x1,y1),(x2,y2),colour,2)
            cv2.circle(output,tuple(int(round(v)) for v in w['floor_point']),5,colour,-1)
            zone=('Location uncertain' if w['zone_ids'] is None else ' / '.join(w['zone_names']) or 'Outside marked zones')
            activity=w.get('activity')
            lines=[f'ID {tid}: {zone}',w['zone_compliance_status']]
            if activity is not None:
                lines.append(f'Activity: {activity}')
            if w['zone_not_detected']:
                lines.append('Not detected: '+', '.join(w['zone_not_detected']))
            base=max(15,min(height-5-len(lines)*17,y1-8))
            for i,text in enumerate(lines):
                cv2.putText(output,text,(max(0,min(width-1,x1)),base+i*17),cv2.FONT_HERSHEY_SIMPLEX,.45,colour,1)
        return output


def export_activity_zones(config,camera_id,frame_size,output_path,
                          workstation_zone_ids=(),walkway_zone_ids=()):
    """Convert saved polygons to the friend's ORIGINAL --zones JSON format.

    This supports running their activity program unchanged as a separate pass.
    It does not merge activity labels into the PPE loop. Specify IDs explicitly;
    a PPE zone is not automatically a workstation or a walkway.
    """
    polygons=scaled_zones(config,camera_id,frame_size)
    chosen=set(workstation_zone_ids)|set(walkway_zone_ids)
    unknown=chosen-set(polygons)
    if unknown:
        raise ValueError(f'Zones not saved for this camera: {sorted(unknown)}')
    converted={'workstations':[polygons[z] for z in workstation_zone_ids],
               'walkways':[polygons[z] for z in walkway_zone_ids]}
    path=Path(output_path)
    if path.suffix.lower()!='.json':
        raise ValueError('Choose a separate .json output filename.')
    # Prevent accidentally replacing the main schema with a legacy export.
    if path.exists():
        existing=json.loads(path.read_text())
        if 'cameras' in existing or 'zone_rules' in existing:
            raise ValueError('Do not overwrite the main zone configuration with an activity export.')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(converted,indent=2),encoding='utf-8')
    return str(path.resolve())
