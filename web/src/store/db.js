// Per-machine baselines and check history live in IndexedDB on the device.
import { openDB } from 'idb'

const dbp = openDB('stetho', 1, {
  upgrade(db) {
    db.createObjectStore('machines', { keyPath: 'id' })
    const c = db.createObjectStore('checks', { keyPath: 'id', autoIncrement: true })
    c.createIndex('byMachine', ['machineId', 'ts'])
  },
})

export const listMachines = async () => (await dbp).getAll('machines')
export const saveMachine = async (m) => (await dbp).put('machines', m)
export const deleteMachine = async (id) => {
  const db = await dbp
  const tx = db.transaction(['machines', 'checks'], 'readwrite')
  await tx.objectStore('machines').delete(id)
  const idx = tx.objectStore('checks').index('byMachine')
  for await (const cur of idx.iterate(IDBKeyRange.bound([id, 0], [id, Infinity]))) await cur.delete()
  await tx.done
}
export const addCheck = async (c) => (await dbp).add('checks', c)
export const machineHistory = async (id) =>
  (await dbp).getAllFromIndex('checks', 'byMachine', IDBKeyRange.bound([id, 0], [id, Infinity]))

if (navigator.storage?.persist) navigator.storage.persist().catch(() => {})
