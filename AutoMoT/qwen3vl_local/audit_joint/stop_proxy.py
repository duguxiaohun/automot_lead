"""Read-only STOP proxy geometry diagnosis; never supplies teacher permission.

The exported actor box alone does not identify a painted stop line or the
original simulator trigger dimensions. Report distances separately, not a
calibrated replacement threshold or an automatically corrected label.
"""
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g


def measure(frame, stop_id):
    stop=next((a for a in frame['actors'] if a['class']=='stop_sign' and a['id']==stop_id),None)
    result=dict(stop_id=stop_id,frame_id=frame['frame_id'],speed_mps=frame['meta'].get('speed'),
                calibration_status='unproven_stop_line',label_authorization=False,
                sources=frame.get('sources',[]))
    if stop is None:
        return dict(result,status='actor_missing')
    ego=next(a for a in frame['actors'] if a['class']=='ego_car')
    box=g.aabb(stop);front=g.aabb(ego)[1]
    x=stop['position'][0]
    result.update(status='geometry_measured',extent=stop['extent'],yaw=stop['yaw'],
                  center_ego=stop['position'],ego_front_x=front,box_aabb=box,
                  center_to_ego_origin_m=x,center_to_ego_front_m=x-front,
                  exported_near_face_to_ego_front_m=box[0]-front,
                  exported_far_face_to_ego_front_m=box[1]-front,
                  longitudinal_box_half_span_m=x-box[0],
                  affects_ego=stop.get('affects_ego'),
                  exported_fixed_1p5_shape=list(stop['extent'])==[1.5,1.5,.5],
                  interpretation='Exported box faces and center are distinct. A negative near-face '
                                 'gap is not proof of crossing a road stop line. Matching a fixed '
                                 'shape does not prove which exporter produced this file.')
    return result
