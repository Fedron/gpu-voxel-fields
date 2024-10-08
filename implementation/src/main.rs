use node::Node;
use utils::Bounds;

mod node;
mod utils;

fn main() {
    let mut node = Node::<Voxel>::new(Bounds {
        min: glam::UVec3::new(0, 0, 0),
        max: glam::UVec3::new(8, 8, 8),
    });

    node.insert(glam::UVec3::new(0, 0, 0), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(1, 0, 0), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(1, 0, 1), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(0, 0, 1), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(0, 1, 0), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(1, 1, 0), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(1, 1, 1), 1, Voxel { color: 0 });
    node.insert(glam::UVec3::new(0, 1, 1), 1, Voxel { color: 0 });

    println!("{:#?}", node);
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Voxel {
    color: u32,
}
