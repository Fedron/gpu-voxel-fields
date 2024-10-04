#[derive(Debug, Clone, Copy)]
pub struct Voxel {
    pub color: [f32; 3],
}

#[derive(Debug, Default)]
pub enum Node {
    #[default]
    Empty,
    Leaf(Voxel),
    Branch(Box<[Node; 8]>),
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Empty, Node::Empty) => true,
            (Node::Leaf(_), Node::Leaf(_)) => true,
            (Node::Branch(_), Node::Branch(_)) => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct Octree {
    root: Node,
    max_depth: u32,
}

impl Octree {
    pub fn new(max_depth: u32) -> Self {
        Self {
            root: Node::Empty,
            max_depth,
        }
    }

    pub fn insert(&mut self, voxel: Voxel, position: glam::UVec3) {
        Self::insert_recursive(&mut self.root, voxel, position, self.max_depth);
    }

    fn insert_recursive(node: &mut Node, voxel: Voxel, position: glam::UVec3, inv_depth: u32) {
        if *node == Node::Empty {
            *node = Node::Branch(Default::default());
        }

        if inv_depth == 0 {
            *node = Node::Leaf(voxel);
            return;
        }

        let min = glam::UVec3::ZERO;
        let max = glam::UVec3::splat(2u32.pow(inv_depth));
        let center = (min + max) / 2;

        let mut octant = 0;
        if position.x >= center.x {
            octant |= 1;
        }
        if position.y >= center.y {
            octant |= 2;
        }
        if position.z >= center.z {
            octant |= 4;
        }

        if let Node::Branch(children) = node {
            children[octant] = Node::Branch(Default::default());
            Self::insert_recursive(&mut children[octant], voxel, position, inv_depth - 1);
        }
    }
}

// use std::{cell::RefCell, rc::Rc};

// #[derive(Debug, Clone)]
// pub struct Node {
//     children: Option<Box<[Option<Rc<RefCell<Node>>>; 8]>>,
//     data: Voxel,
// }

// #[derive(Debug)]
// pub struct Octree {
//     root: Option<Rc<RefCell<Node>>>,
//     size: u32,
//     max_depth: u32,
// }

// impl Octree {
//     pub fn new(size: u32, max_depth: u32) -> Self {
//         Self {
//             root: None,
//             size,
//             max_depth,
//         }
//     }

//     pub fn insert(&mut self, voxel: Voxel, position: glam::UVec3) {
//         self.insert_recursive(&mut self.root.clone(), voxel, position, 0);
//     }

//     fn insert_recursive(
//         &mut self,
//         node: &mut Option<Rc<RefCell<Node>>>,
//         voxel: Voxel,
//         position: glam::UVec3,
//         depth: u32,
//     ) {
//         if node.is_none() {
//             println!("hello");
//             *node = Some(Rc::new(RefCell::new(Node {
//                 children: None,
//                 data: voxel,
//             })));
//         }

//         let node = node.as_ref().unwrap();
//         {
//             let mut node = node.borrow_mut();
//             node.data = voxel;
//         }

//         if depth == self.max_depth {
//             return;
//         }

//         let size = self.size / 2u32.pow(depth);
//         let child_position = glam::uvec3(
//             (size * position.x) + (size / 2),
//             (size * position.x) + (size / 2),
//             (size * position.x) + (size / 2),
//         );

//         let child_index =
//             (child_position.x << 0) | (child_position.y << 1) | (child_position.z << 2);

//         let position = glam::uvec3(
//             (position.x << 1) | child_position.x,
//             (position.y << 1) | child_position.y,
//             (position.z << 1) | child_position.z,
//         );

//         self.insert_recursive(
//             &mut node
//                 .borrow()
//                 .children
//                 .as_ref()
//                 .and_then(|children| children[child_index as usize].clone()),
//             voxel,
//             position,
//             depth + 1,
//         );
//     }
// }
