use tf_demo_parser::demo::message::packetentities::EntityId;

// Compute the difference in viewangles. We have to account for the fact viewangles are in a circle.
// E.g. If viewangle goes from 350 to 10 degrees, we want to return 20 degrees.
pub fn viewangle_delta(curr_viewangle: f32, curr_pitchangle: f32, prev_viewangle: f32, prev_pitchangle: f32, tick_delta: u32) -> (f32, f32) {
    let tick_delta = if tick_delta < 1 { 1 } else { tick_delta };
    let va_delta = {
        let diff = (curr_viewangle - prev_viewangle).rem_euclid(360.0);
        if diff > 180.0 {
            diff - 360.0
        } else {
            diff
        }
    } / tick_delta as f32;
    let pa_delta = (curr_pitchangle - prev_pitchangle) / tick_delta as f32;
    (va_delta, pa_delta)
}

// Since TF2 has an object limit of 2048, the lowest 11 bits of the handle ID represent the entity ID.
// Source: https://developer.valvesoftware.com/wiki/CHandle
#[allow(dead_code)]
pub fn handle_to_entid(handle: u32) -> EntityId {
    let entid = handle & 0x7FF;
    EntityId::from(entid)
}