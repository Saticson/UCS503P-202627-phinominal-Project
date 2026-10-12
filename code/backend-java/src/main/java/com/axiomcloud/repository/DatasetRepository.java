package com.axiomcloud.repository;

import com.axiomcloud.model.DatasetEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface DatasetRepository extends JpaRepository<DatasetEntity, Integer> {

    List<DatasetEntity> findAllByOrderByCreatedAtDesc();

    Optional<DatasetEntity> findByIsExampleTrueAndName(String name);
}
