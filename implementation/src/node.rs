use core::fmt::Debug;

use crate::utils::Bounds;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum Octant {
    BottomLeftBack = 0,
    BottomRightBack = 1,
    BottomRightFront = 2,
    BottomLeftFront = 3,
    TopLeftBack = 4,
    TopRightBack = 5,
    TopRightFront = 6,
    TopLeftFront = 7,
}

impl TryFrom<u8> for Octant {
    // TODO: Use actual error type
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Octant::BottomLeftBack),
            1 => Ok(Octant::BottomRightBack),
            2 => Ok(Octant::BottomRightFront),
            3 => Ok(Octant::BottomLeftFront),
            4 => Ok(Octant::TopLeftBack),
            5 => Ok(Octant::TopRightBack),
            6 => Ok(Octant::TopRightFront),
            7 => Ok(Octant::TopLeftFront),
            _ => Err(()),
        }
    }
}

impl Octant {
    /// Returns the 3D position of the child octant relative to the parent octant.
    fn offset(&self) -> glam::UVec3 {
        match self {
            Octant::BottomLeftBack => glam::uvec3(0, 0, 0),
            Octant::BottomRightBack => glam::uvec3(1, 0, 0),
            Octant::BottomRightFront => glam::uvec3(1, 0, 1),
            Octant::BottomLeftFront => glam::uvec3(0, 0, 1),
            Octant::TopLeftBack => glam::uvec3(0, 1, 0),
            Octant::TopRightBack => glam::uvec3(1, 1, 0),
            Octant::TopRightFront => glam::uvec3(1, 1, 1),
            Octant::TopLeftFront => glam::uvec3(0, 1, 1),
        }
    }

    /// Given a midpoint of an octant, and a position within that octant, returns the child octant the position would reside in.
    fn from_position(position: glam::UVec3, midpoint: glam::UVec3) -> Self {
        if position.z < midpoint.z {
            if position.y < midpoint.y {
                if position.x < midpoint.x {
                    Octant::BottomLeftBack
                } else {
                    Octant::BottomRightBack
                }
            } else {
                if position.x < midpoint.x {
                    Octant::TopLeftBack
                } else {
                    Octant::TopRightBack
                }
            }
        } else {
            if position.y < midpoint.y {
                if position.x < midpoint.x {
                    Octant::BottomLeftFront
                } else {
                    Octant::BottomRightFront
                }
            } else {
                if position.x < midpoint.x {
                    Octant::TopLeftFront
                } else {
                    Octant::TopRightFront
                }
            }
        }
    }
}

struct NodeInfo {
    resolution: u32,
    octant: Octant,
}

#[derive(Debug, Clone, Default)]
enum NodeType<T>
where
    T: Debug + Clone + Copy + PartialEq,
{
    Leaf(T),
    Branch([Box<Option<Node<T>>>; 8]),
    #[default]
    Empty,
}

#[derive(Debug, Clone)]
pub struct Node<T>
where
    T: Debug + Clone + Copy + PartialEq,
{
    ty: NodeType<T>,
    bounds: Bounds,
}

impl<T> Node<T>
where
    T: Debug + Clone + Copy + PartialEq,
{
    /// Creates a new empty node with the specified bounds.
    pub fn new(bounds: Bounds) -> Self {
        Self {
            ty: NodeType::Empty,
            bounds,
        }
    }

    /// Inserts new data at the position specified.
    ///
    /// Creates new nodes as necessary to contain the position.
    pub fn insert(&mut self, position: glam::UVec3, min_resolution: u32, data: T) {
        // If the position is not within the bounds of this node, return
        if !self.bounds.contains(position) {
            return;
        }

        // We are at the minimum resolution and cannot subdivide further, so become a leaf node
        if self.resolution() == min_resolution {
            self.ty = NodeType::Leaf(data);
            return;
        }

        let child_info = self.child_node_info(position).unwrap();
        let child_bounds = self.child_bounds(&child_info);

        match self.ty {
            // If the node is a leaf, we need to convert it to a branch node.
            // All octants will be the same as the current leaf node, except for the one we are inserting into (which will remain `None`).
            NodeType::Leaf(data) => {
                let mut branch: [Box<Option<Node<T>>>; 8] =
                    core::array::from_fn(|_| Box::new(None));
                for i in 0..8 {
                    if i != child_info.octant as u8 {
                        let new_octant = Octant::try_from(i).unwrap();
                        let new_bounds = self.child_bounds(&NodeInfo {
                            resolution: child_info.resolution,
                            octant: new_octant,
                        });

                        let mut new_node = Node::new(new_bounds);
                        new_node.ty = NodeType::Leaf(data.clone());

                        branch[i as usize] = Box::new(Some(new_node));
                    }
                }
            }
            // Node is already a branch, so we can just recursivly insert into the child node
            NodeType::Branch(ref mut branch) => {
                let mut child_node = if branch[child_info.octant as usize].as_ref().is_some() {
                    branch[child_info.octant as usize].take().unwrap()
                } else {
                    Node::new(child_bounds)
                };

                child_node.insert(position, min_resolution, data);
                branch[child_info.octant as usize] = Box::new(Some(child_node));

                self.compact();
            }
            // Node is empty, so we can just insert a new node recursively
            NodeType::Empty => {
                let mut new_node = Node::new(child_bounds);
                new_node.insert(position, min_resolution, data);

                self.ty = NodeType::Branch(core::array::from_fn(|i| {
                    if i == child_info.octant as usize {
                        Box::new(Some(new_node.clone()))
                    } else {
                        Box::new(None)
                    }
                }));
            }
        }
    }

    /// Where possible merges branch nodes with identical leaf nodes into a single leaf node.
    fn compact(&mut self) {
        if !matches!(self.ty, NodeType::Branch(_)) {
            return;
        }

        let children = match &mut self.ty {
            NodeType::Branch(children) => children,
            _ => unreachable!(),
        };

        let mut target_data = None;
        for i in 0..8 {
            if let Some(child_node) = children[i].as_ref() {
                match &child_node.ty {
                    NodeType::Leaf(voxel) => match target_data {
                        Some(data) if data != *voxel => return,
                        None => target_data = Some(*voxel),
                        _ => {}
                    },
                    _ => return,
                }
            } else {
                return;
            }
        }

        if let Some(target_data) = target_data {
            self.ty = NodeType::Leaf(target_data);
        }
    }

    fn resolution(&self) -> u32 {
        self.bounds.max.x - self.bounds.min.x
    }

    fn child_node_info(&self, position: glam::UVec3) -> Option<NodeInfo> {
        if self.bounds.contains(position) {
            let resolution = self.resolution() / 2;
            let curr_midpoint = self.bounds.min + glam::UVec3::splat(resolution);

            Some(NodeInfo {
                resolution,
                octant: Octant::from_position(position, curr_midpoint),
            })
        } else {
            None
        }
    }

    fn child_bounds(&self, child_info: &NodeInfo) -> Bounds {
        let resolution = glam::UVec3::splat(child_info.resolution);
        let min = self.bounds.min + (resolution * child_info.octant.offset());
        let max = min + resolution;

        Bounds { min, max }
    }
}
