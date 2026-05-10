import { createContext, useContext } from 'react';

export const SubNavContext = createContext({ active: null, setActive: () => {} });

export const useSubNav = () => useContext(SubNavContext);
