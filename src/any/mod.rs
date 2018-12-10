use std::any::Any;
use std::cmp::Ord;

pub struct AnyVal {
    val: Box<dyn Any>
}

pub struct AnyVec {
    vec: Box<dyn Any>
}

impl AnyVal {
    pub fn eq<T: 'static + Eq>(&self, other: &T) -> bool {
        self.val.downcast_ref::<T>().filter(|&v| v.eq(other)).is_some()
    }

    pub fn lt<T: 'static + Ord>(&self, other: &T) -> bool {
        self.val.downcast_ref::<T>().filter(|&v| v < other).is_some()
    }

    pub fn gt<T: 'static + Ord>(&self, other: &T) -> bool {
        self.val.downcast_ref::<T>().filter(|&v| v > other).is_some()
    }
}

impl AnyVec {
    pub fn new<T: 'static>(vec: Vec<T>) -> AnyVec {
        AnyVec {
            vec: Box::new(vec)
        }
    }

    pub fn get<'a, T: 'static>(&'a self) -> Option<&'a Vec<T>> {
        self.vec.downcast_ref::<Vec<T>>()
    }

    pub fn get_item<'a, T: 'static>(&'a self, index: usize) -> Option<&T> {
        self.get::<T>().map(|v| &v[index])
    }
}
